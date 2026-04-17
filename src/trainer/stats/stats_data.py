import src.trainer.stats.utils as utils
import pynvml
import src.config as config
import datetime
import pandas as pd
import os
from abc import ABC, abstractmethod

# utility classes to collect and save the data 
class StatsData(ABC):
    def __init__(self, conf: config.Config):
        self.conf = conf

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def create_df(self):
        pass    

    def to_csv(self, exp, dir, step):
        """
        exp: the name of the experiment: time, resource, etc.
        dir: folder where we store the data
        step: train, step, forward, backward, optim
        """
        df = self.create_df()

        # build file name with params
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

        full_filename = f"{time}_{exp}-{step}" + full_filename + ".csv"

        df.to_csv(os.path.join(os.getcwd(), "final_data_analysis", dir, step, full_filename), index=False)

class TimingStatsData(StatsData):
    """
    Class to store the data for each resource stat we wish to collect
    - time taken for that step
    """
    def __init__(self, conf: config.Config):
        super().__init__(conf)
        self.times = utils.RunningTimer()

    def start(self):
        self.times.start()
        
    def stop(self):
        self.times.stop()        

    def create_df(self):
        df = pd.DataFrame({"time": self.times.stat.history})
        return df

# CPU/GPU/memory usage
class ResourceStatsData(StatsData):
    """
    Class to store the data for each resource stat we wish to collect
    - time taken for that step
    - GPU utilization
    - CPU utilization
    """
    def __init__(self, conf, GPU_handle, CPU_process, device):
        super().__init__(conf)
        self.handle = GPU_handle
        self.process = CPU_process
        self.device = device

        self.times = utils.RunningTimer() #duration of each step
        self.gpu_util = utils.RunningStat()
        self.cpu_util = utils.RunningStat()
        self.gpu_mem = utils.RunningStat()
        self.timestamps = utils.RunningTimer() # timestamps
        self.timestamps.start()

    def start(self):
        self.times.start()
        self.process.cpu_percent() # note that if fwd/bkwd is True, this will not be accurate for the step and should be run separately. 
        
    def stop(self):
        self.times.stop()
        self.gpu_util.update(value = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
        self.cpu_util.update(value = self.process.cpu_percent()) # the sum of utilization
        self.gpu_mem.update(value = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
        self.timestamps.stop()

    def create_df(self):
        df = pd.DataFrame({"iteration": list(range(1, len(self.times.stat.history)+1)),
                    "timestamp": self.timestamps.stat.history,
                    "duration": self.times.stat.history, # duration of each step
                    "GPU utilization": self.gpu_util.history,
                    "CPU utilization": self.cpu_util.history,
                    "GPU memory": self.gpu_mem.history,
                    })

        return df