import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import datetime
import logging
import torch
import psutil
import os
import pandas as pd
import pynvml
import time

logger = logging.getLogger(__name__)
trainer_stats_name="resource_usage"

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
    - GPU utilization
    - CPU utilization
    """
    def __init__(self, GPU_handle, CPU_process, device, conf: config.Config):
        self.handle = GPU_handle
        self.process = CPU_process
        self.device = device
        self.conf = conf

        self.times = utils.RunningTimer()
        self.gpu_util = utils.RunningStat()
        self.cpu_util = utils.RunningStat()
        self.gpu_mem = utils.RunningStat()

        # to remove if bad
        self.cpu_user = utils.RunningStat()
        self.cpu_system = utils.RunningStat()
        #self.cpu_child_user = utils.RunningStat()
        #self.cpu_child_system = utils.RunningStat()

        self.cpu_idle = utils.RunningStat()

    def start(self):
        self.times.start()
        
    def stop(self):
        self.times.stop()
        self.gpu_util.update(value = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
        #self.cpu_util.update(value = self.process.cpu_percent()) # may not include worker because the worker is not the same process
        self.cpu_util.update(value = psutil.cpu_percent()) # system wide, includes the process -- check parent child process.
        
        cpu_times = psutil.cpu_times()
        self.cpu_user.update(value = cpu_times.user)
        self.cpu_system.update(value = cpu_times.system)
        self.cpu_idle.update(value = cpu_times.idle)
        #self.cpu_child_user.update(value = cpu_times.children_user)
        #self.cpu_child_system.update(value = cpu_times.children_system)

        #self.gpu_mem.update(value = torch.cuda.memory_allocated(self.device))
        self.gpu_mem.update(value = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)


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
        print("Time for step")
        self.log(self.times.stat)
        print("GPU utilization")
        self.log(self.gpu_util)
        print("CPU utilization")
        self.log(self.cpu_util)
        print("GPU memory")
        self.log(self.gpu_mem)

    def to_csv(self, step):
        print(f"time: {len(self.times.stat.history)}")
        print(f"GPU utilization: {len(self.gpu_util.history)}")
        print(f"CPU utilization: {len(self.cpu_util.history)}")
        print(f"GPU memory: {len(self.gpu_mem.history)}")
        print(f"CPU user time: {len(self.cpu_user.history)}")
        print(f"CPU system time: {len(self.cpu_system.history)}")
        print(f"CPU idle time: {len(self.cpu_idle.history)}")

        df = pd.DataFrame({"iteration": list(range(1, len(self.times.stat.history)+1)),
                            "time": self.times.stat.history,
                            "GPU utilization": self.gpu_util.history,
                            "CPU utilization": self.cpu_util.history,
                            "GPU memory": self.gpu_mem.history,
                            "CPU user time": self.cpu_user.history,
                            "CPU system time": self.cpu_system.history,
                            "CPU idle time": self.cpu_idle.history
                            #"CPU children user time": self.cpu_child_user.history,
                            #"CPU children system time": self.cpu_child_system.history
                            })
        

        # build file name with params
        time = datetime.datetime.now().strftime("%m-%d-%H-%M")
        exp = "resource"
        params = {"batch": self.conf.batch_size,
                  "work": self.conf.data_configs.whisper_data.num_workers,
                  "samples": self.conf.data_configs.whisper_data.num_samples, 
                  "repeat": self.conf.data_configs.whisper_data.repeat,
                  "labels": self.conf.data_configs.whisper_data.num_labels}
        full_filename = ""
        for param in params:
            full_filename += f"_{param}-{params[param]}"
        full_filename = f"{time}_{exp}-{step}" + full_filename + ".csv"

        df.to_csv(os.path.join(os.getcwd(), "final_data_analysis", "resource_data", full_filename), index=False)

class ResourceStats(base.TrainerStats):

    def __init__(self, device: torch.device, conf: config.Config):
        super().__init__()
        self.device = device

        # NVML initialization step
        pynvml.nvmlInit()

        # NVML get the GPU device
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # create the CPU process thing for CPU measurements
        self.process = psutil.Process()

        # create the data storages
        self.step_data = ResourceStatsData(self.handle, self.process, self.device, conf)
        self.fwd_data = ResourceStatsData(self.handle, self.process, self.device, conf)
        self.bkwd_data = ResourceStatsData(self.handle, self.process, self.device, conf)
        self.optim_data = ResourceStatsData(self.handle, self.process, self.device, conf)

        # remove for submission
        self.get_step = True
        self.get_fwd = False
        self.get_bkwd = False
        self.get_optim = False

        self.n_steps = 0

        # manually set rate according to batch size, so it's collected ~ 1-2 seconds
        '''
        if conf.batch_size == 32:
            self.measurement_rate = 3
        elif conf.batch_size == 64:
            self.measurement_rate = 2
        elif conf.batch_size == 16:
            self.measurement_rate = 5
        else:
            self.measurement_rate = 1
        '''
        self.measurement_rate = 1
        self.start = 0


    def start_train(self) -> None:
        """Start training.

        This method should be called by trainers when starting the training loop.

        """
        self.start = time.time()

    def stop_train(self) -> None:
        """Stop training.

        This method should be called by trainers when the training is done.

        """
        end = time.time() - self.start
        print(f"Total training time: {end//60} minutes and {round(end%60, 2)} seconds")


    def start_step(self):
        if self.get_step and self.n_steps % self.measurement_rate == 0:
            torch.cuda.synchronize(self.device)
            self.step_data.start()
        
    def stop_step(self):
        self.n_steps += 1
        if self.get_step and self.n_steps % self.measurement_rate == 0:
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




