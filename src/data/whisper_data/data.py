import datasets
import src.config as config
import torch.utils.data 
from transformers import WhisperProcessor

data_load_name="whisper_data"

# generate one random piece of data
def generate_random_audio(extractor):
    wav = list(torch.rand(10000) * 2 - 1)
    data = extractor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
    return data["input_features"][0] # this is the spectrogram form of the random audio data

# generate one random label
def generate_random_label(num_labels):
    num_labels = num_labels
    label = torch.randint(0, num_labels, ())
    return label

# create some synthetic data
class SyntheticDataWhisper(torch.utils.data.Dataset):
    '''
    Class for creating our own synthetic dataset.

    The data attribute is a list of dicts, where the dicts contain an audio and a random label.
    '''
    def __init__(self, n, repeat, num_labels):
        self.n = n # batch size?
        self.repeat = repeat
        whisper_extractor = WhisperProcessor.from_pretrained("openai/whisper-tiny") # converts audio to spectrogram data to be used in the model
        self.data = [self.generate_random_sample(whisper_extractor, num_labels) for _ in range(n)] # generates the data

    def generate_random_sample(self, extractor, num_labels):
        '''
        Returns a dictionary mapping sort of model names (or something similar)
        to the function that generates the model data.clear

        gen() is a dict containing the data/label functions (in the case for audio)
        '''
        return {"input_features": generate_random_audio(extractor),
                "labels": generate_random_label(num_labels)}

    def __getitem__(self, i):
        return self.data[i % self.n]

    def __len__(self):
        return self.n * self.repeat


# return synthetic data obj
def load_data(conf : config.Config) -> torch.utils.data.Dataset:
    """Simple function to load a dataset based on the provided config object.
    """
    '''
    train_files = None
    if conf.data_configs.dataset.train_files is not None and conf.data_configs.dataset.train_files != "":
        train_files = {"train": conf.data_configs.dataset.train_files}
    return datasets.load_dataset(conf.data_configs.dataset.name, data_files=train_files, split=conf.data_configs.dataset.split, num_proc=conf.data_configs.dataset.load_num_proc)
    '''
    return SyntheticDataWhisper(n=conf.batch_size, repeat=conf.data_configs.whisper_data.num_samples, num_labels=2) #conf._arg_batch_size, conf.model_configs.num_labels