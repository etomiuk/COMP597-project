import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import time
import logging
import torch
import psutil
import os
import pandas as pd
import pynvml
import datetime

logger = logging.getLogger(__name__)
trainer_stats_name="timing"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    # Handle additional configurations here
    # used the same code as the simple stats 
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to simple trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return ResourceStats(device=device, conf=conf)

class ResourceStatsData():
    """
    Class to store the data for each resource stat we wish to collect
    - time taken for that step
    """
    def __init__(self, conf: config.Config):
        self.conf = conf
        self.times = utils.RunningTimer()

    def start(self):
        self.times.start()
        
    def stop(self):
        self.times.stop()


    @staticmethod
    def log(data_to_log):
        """ Same as log_analysis in utils file, but did does not divide
        The method is copied here because did not want to change the starter code
        """
        data = torch.tensor(data_to_log.history)
        data = data.to(torch.float)
        print(f"mean   : {data.mean() : .4f}")
        print(f"q0.001 : {data.quantile(q=torch.tensor(0.001), interpolation='nearest') : .4f}")
        print(f"q0.01  : {data.quantile(q=torch.tensor(0.010), interpolation='nearest') : .4f}")
        print(f"q0.1   : {data.quantile(q=torch.tensor(0.100), interpolation='nearest') : .4f}")
        print(f"q0.25  : {data.quantile(q=torch.tensor(0.250), interpolation='nearest') : .4f}")
        print(f"q0.5   : {data.quantile(q=torch.tensor(0.500), interpolation='nearest') : .4f}")
        print(f"q0.75  : {data.quantile(q=torch.tensor(0.750), interpolation='nearest') : .4f}")
        print(f"q0.9   : {data.quantile(q=torch.tensor(0.900), interpolation='nearest') : .4f}")
        print(f"q0.99  : {data.quantile(q=torch.tensor(0.990), interpolation='nearest') : .4f}")
        print(f"q0.999 : {data.quantile(q=torch.tensor(0.999), interpolation='nearest') : .4f}")

    def print_stats(self):
        print("Time")
        self.log(self.times.stat)

    def to_csv(self, step):
        df = pd.DataFrame({"iteration": list(range(1, len(self.times.stat.history)+1)),
                            "time": self.times.stat.history,
                            })

        # build file name with params
        time = datetime.datetime.now().strftime("%m-%d-%H-%M")
        exp = "time"
        params = {"batch": self.conf.batch_size,
                  "work": self.conf.data_configs.whisper_data.num_workers,
                  "samples": self.conf.data_configs.whisper_data.num_samples, 
                  "repeat": self.conf.data_configs.whisper_data.repeat,
                  "labels": self.conf.data_configs.whisper_data.num_labels}
        full_filename = ""
        for param in params:
            full_filename += f"_{param}-{params[param]}"
        full_filename = f"{time}_{exp}-{step}" + full_filename + ".csv"

        df.to_csv(os.path.join(os.getcwd(), "final_data_analysis", "timing_data", full_filename), index=False)

class ResourceStats(base.TrainerStats):

    def __init__(self, device: torch.device, conf: config.Config):
        super().__init__()
        self.device = device

        # create the data storages
        self.train_data = ResourceStatsData(conf)
        self.step_data = ResourceStatsData(conf)
        self.fwd_data = ResourceStatsData(conf)
        self.bkwd_data = ResourceStatsData(conf)
        self.optim_data = ResourceStatsData(conf)

        # remove for submission
        self.get_step = True
        self.get_fwd = False
        self.get_bkwd = False
        self.get_optim = False


    def start_train(self) -> None:
        """Start training.

        This method should be called by trainers when starting the training loop.

        """
        torch.cuda.synchronize(self.device)
        self.train_data.start()

    def stop_train(self) -> None:
        """Stop training.

        This method should be called by trainers when the training is done.

        """
        torch.cuda.synchronize(self.device)
        self.train_data.stop()


    def start_step(self):
        if self.get_step:
            torch.cuda.synchronize(self.device)
            self.step_data.start()
        
    def stop_step(self):
        if self.get_step:
            torch.cuda.synchronize(self.device)
            self.step_data.stop()

    def start_forward(self) -> None:
        """Start the forward pass.

        This method should be called by trainers at the beginning of every 
        forward pass.

        """
        if self.get_fwd:
            torch.cuda.synchronize(self.device)
            self.fwd_data.start()

    def stop_forward(self) -> None:
        """Stop the forward pass.

        This method should be called by trainers at the end of every forward 
        pass.

        """
        if self.get_fwd:
            torch.cuda.synchronize(self.device)
            self.fwd_data.stop()
        

    def log_loss(self, loss: torch.Tensor) -> None:
        """Logs the loss of the current step by passing it to the stats.

        """
        pass

    def start_backward(self) -> None:
        """Start the backward pass.

        This method should be called by trainers at the start of every backward 
        pass.

        """
        if self.get_bkwd:
            torch.cuda.synchronize(self.device)
            self.bkwd_data.times.start()
        

    def stop_backward(self) -> None:
        """Stop the backward pass

        This method should be called by trainers at the end of every backward 
        pass.

        """
        if self.get_bkwd:
            torch.cuda.synchronize(self.device)
            self.bkwd_data.stop()
        

    def start_optimizer_step(self) -> None:
        """Start the optimizer step.

        This method should be called by trainers at the start of the optimizer 
        step.

        """
        if self.get_optim:
            torch.cuda.synchronize(self.device)
            self.optim_data.start()


    def stop_optimizer_step(self) -> None:
        """Stop the optimizer step.

        This method should be called by trainers at the end of the optimizer 
        step.

        """
        if self.get_optim:
            torch.cuda.synchronize(self.device)
            self.optim_data.stop()

    def start_save_checkpoint(self) -> None:
        """Start checkpointing.

        This method should be called by trainers when they initiate a 
        checkpointing step.

        """
        pass

    def stop_save_checkpoint(self) -> None:
        """Stop checkpointing.

        This method should be called by trainers at the end of a checkpointing 
        step.

        """
        pass

    def log_step(self) -> None:
        """Logs information about the previous step.

        This method should be called after the `stop_step`. It should log 
        information about the previous training step.

        """
        pass

    def log_stats(self):
        if self.get_step:
            #self.step_data.print_stats()
            self.step_data.to_csv(step="step")
        if self.get_fwd:
            #self.fwd_data.print_stats()
            self.fwd_data.to_csv(step="forward")
        if self.get_bkwd:
            #self.bkwd_data.print_stats()
            self.bkwd_data.to_csv(step="backward")
        if self.get_optim:
            #self.optim_data.print_stats()
            self.optim_data.to_csv(step="optim")




