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
trainer_stats_name="timing_minimal"

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

    def print_stats(self):
        print(f"Total time: {self.times.stat.history[0]/1e6} ms -- {self.times.stat.history[0]/1e9} sec")
        

class ResourceStats(base.TrainerStats):

    def __init__(self, device: torch.device, conf: config.Config):
        super().__init__()
        self.device = device

        # create the data storages
        self.train_data = ResourceStatsData(conf)


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
        pass
        
    def stop_step(self):
        pass

    def start_forward(self) -> None:
        """Start the forward pass.

        This method should be called by trainers at the beginning of every 
        forward pass.

        """
        pass

    def stop_forward(self) -> None:
        """Stop the forward pass.

        This method should be called by trainers at the end of every forward 
        pass.

        """
        pass
        

    def log_loss(self, loss: torch.Tensor) -> None:
        """Logs the loss of the current step by passing it to the stats.

        """
        pass

    def start_backward(self) -> None:
        """Start the backward pass.

        This method should be called by trainers at the start of every backward 
        pass.

        """
        pass
        

    def stop_backward(self) -> None:
        """Stop the backward pass

        This method should be called by trainers at the end of every backward 
        pass.

        """
        pass
        

    def start_optimizer_step(self) -> None:
        """Start the optimizer step.

        This method should be called by trainers at the start of the optimizer 
        step.

        """
        pass


    def stop_optimizer_step(self) -> None:
        """Stop the optimizer step.

        This method should be called by trainers at the end of the optimizer 
        step.

        """
        pass

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
        self.train_data.print_stats()





