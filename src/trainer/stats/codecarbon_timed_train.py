from codecarbon import OfflineEmissionsTracker
from codecarbon.external.logger import logger
import codecarbon
import codecarbon.core.cpu 
import logging
import os
import src.config as config
import src.trainer.stats.base as base
import torch
import src.trainer.stats.stats_data as stats

logger = logging.getLogger(__name__)

# artificially force psutil to fail, so that CodeCarbon uses constant mode for CPU measurements
codecarbon.core.cpu.is_psutil_available = lambda: False

trainer_stats_name="codecarbon_timed_train"

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
        
        # Normal-mode tracker to track the entire training loop
        self.total_training_tracker = OfflineEmissionsTracker(
            project_name = project_name, 
            country_iso_code = "CAN",
            region = "quebec",
            save_to_file = False, 
            output_handlers = [], # no need to save the codecarbon outputs for this experiment
            allow_multiple_runs = True,
            log_level = "warning",
            gpu_ids = [gpu_id],
        )

        # timer
        self.train_data = stats.TimingStatsData(conf)

    def start_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.train_data.start() # timer
        self.total_training_tracker.start() # CodeCarbon

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        self.total_training_tracker.stop() # CodeCarbon
        self.train_data.stop() # timer

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass
        

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
        self.train_data.to_csv(exp="overhead", dir="overhead", step="e2e_cc")

    def log_loss(self, loss: torch.Tensor) -> None:
        """
        Take the loss from the training loop and log it to the CodeCarbon tracker file.
        """
        pass

