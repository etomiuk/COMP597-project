from codecarbon import OfflineEmissionsTracker
from codecarbon.core.util import backup
from codecarbon.external.logger import logger
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData, TaskEmissionsData
import codecarbon
import codecarbon.core.cpu 
import logging
import os
import pandas as pd
import src.config as config
import src.trainer.stats.base as base
import torch
import datetime
import src.trainer.stats.stats_data as stats
from src.trainer.stats.codecarbon import SimpleFileOutput

logger = logging.getLogger(__name__)
trainer_stats_name="codecarbon_timed_step"

# artificially force psutil to fail, so that CodeCarbon uses constant mode for CPU measurements
codecarbon.core.cpu.is_psutil_available = lambda: False

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to codecarbon trainer stats. Using default PyTorch device")
        device = torch.get_default_device() 
    return CodeCarbonStats(device, conf.trainer_stats_configs.codecarbon.run_num, conf.trainer_stats_configs.codecarbon.project_name, conf.trainer_stats_configs.codecarbon.output_dir, conf)

class CodeCarbonStats(base.TrainerStats):
    """Provides energy consumed and carbon emitted during model training. 
    
    This class measures the energy consumption and carbon emissions of the 
    forward pass, backward pass, and optimiser step, as well as of the training 
    as a whole.

    Implemented using the CodeCarbon library: 
    https://mlco2.github.io/codecarbon/.

    Parameters
    ----------
    device
        A PyTorch device which will be the targets of the measurements.
    run_num
        Used to number different experiments in case their measurements get 
        merged into a single file.
    project_name
        Used by CodeCarbon to identify the experiments. 

    """

    def __init__(self, device : torch.device, run_num : int, project_name : str, output_dir : str, conf: config.Config) -> None: 
        
        # Track current iteration number in the training loop
        self.iteration = 0
        
        # CUDA device indicates the current GPU assigned to this process (0, 1, 2, ...)
        self.device = device
        
        # tracking the run number to distinguish between different parameter settings
        self.run_num = run_num

        # GPU ranks - wrap in torch.device
        gpu_id = self.device.index
        
        # log the losses
        self.losses = []
        self.project_name = project_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.conf = conf

        ### PARAMS FOR FILE NAME ###
        time = datetime.datetime.now().strftime("%m-%d-%H-%M")
        params = {"batch": self.conf.batch_size,
                  "work": self.conf.data_configs.whisper_data.num_workers,
                  "samples": self.conf.data_configs.whisper_data.num_samples, 
                  "repeat": self.conf.data_configs.whisper_data.repeat,
                  "labels": self.conf.data_configs.whisper_data.num_labels}
        
        full_filename = ""
        for param in params:
            full_filename += f"_{param}-{params[param]}"

        # add whether data is loaded on the fly or before training to the file name
        if self.conf.data_configs.whisper_data.onfly == 'y':
            full_filename += "_onfly"
        else:
            full_filename += "_preload"

        full_filename = f"{time}_{full_filename}"

        file_path = os.path.join(os.getcwd(), "final_data_analysis", "energy_data", full_filename)

        # Normal-mode tracker to track the entire training loop
        self.total_training_tracker = OfflineEmissionsTracker(
            project_name = project_name, 
            country_iso_code = "CAN",
            region = "quebec",
            save_to_file = False, 
            output_handlers = [SimpleFileOutput(output_file_name = f"{file_path}_cc_full_rank_{gpu_id}.csv", output_dir=output_dir)],
            allow_multiple_runs = True,
            log_level = "warning",
            gpu_ids = [gpu_id],
        )

        # Task-mode tracker to track steps (iterations) within the training loop
        self.training_step_tracker = OfflineEmissionsTracker(
            project_name = project_name, 
            experiment_name = "steps", #experiment_name required by task_out() method
            country_iso_code = "CAN", 
            region = "quebec", 
            save_to_file = False, 
            output_handlers = [SimpleFileOutput(output_file_name = f"{file_path}_cc_step_rank_{gpu_id}.csv", output_dir=output_dir)],
            allow_multiple_runs = True, 
            api_call_interval = -1, 
            gpu_ids = [gpu_id],
            log_level = "warning",
        )

        # Initialise task-mode trackers
        self.training_step_tracker.start() 

        # training timer
        self.train_timer = stats.TimingStatsData(conf)

    def start_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.train_timer.start()
        self.total_training_tracker.start()

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.total_training_tracker.stop()
        self.training_step_tracker.stop()
        self.train_timer.stop()

    def start_step(self) -> None:
        self.iteration += 1
        if self.iteration%2 == 1: # take every two iterations starting at iteration 1
            torch.cuda.synchronize(self.device)
            self.training_step_tracker.start_task(task_name = f"Step #{self.iteration}")

    def stop_step(self) -> None:
        if self.iteration%2 == 0: # we only stop the measurement, once the iteration has increased 
            torch.cuda.synchronize(self.device)
            self.training_step_tracker.stop_task(task_name = f"Step #{self.iteration-1}")
        

    def start_forward(self) -> None: 
        pass

    def stop_forward(self) -> None: 
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        """
        Log the loss statistics to an external file.
        """
        # losses as dataframe
        df = pd.DataFrame([[x["task_name"], x["loss"].item()] for x in self.losses])
        
        # save to file ({output_dir}/losses/run_{run_num}_cc_loss_rank_{gpu_id}.csv)
        run_number = f"run_{self.run_num}_"
        gpu_id = self.device.index
        losses_dir = os.path.join(self.output_dir, "losses")
        os.makedirs(losses_dir, exist_ok=True)
        save_file_path = os.path.join(losses_dir, f"{run_number}cc_loss_rank_{gpu_id}.csv")
        df.to_csv(save_file_path, index=False)

        logger.info(f"CODECARBON LOSS LOGGING: Rank {gpu_id} - Run {self.run_num} - Losses saved to {save_file_path}")

        # save timing data to the overhead 
        self.train_timer.to_csv(exp="overhead", dir="overhead", step="fine_cc")


    def log_loss(self, loss: torch.Tensor) -> None:
        """
        Take the loss from the training loop and log it to the CodeCarbon tracker file.
        """
        pass

