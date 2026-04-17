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
    def __init__(self, conf):
        
        # set params relating to nb of samples & labels
        self.n = conf.data_configs.whisper_data.num_samples # number of unique samples
        self.repeat = conf.data_configs.whisper_data.repeat # number of times samples are repeated
        self.num_labels = conf.data_configs.whisper_data.num_labels

        # preprocessor
        self.whisper_extractor = WhisperProcessor.from_pretrained("openai/whisper-tiny") # converts audio to spectrogram data to be used in the model

        # determine data loading mode
        self.onfly = conf.data_configs.whisper_data.onfly
        if self.onfly == 'y':
            print("Data loading on the fly")
        else:
            print("Data loading before training")
            self.data = [self.generate_random_sample(self.whisper_extractor, self.num_labels) for _ in range(self.n)] # generates the data

    def generate_random_sample(self, extractor, num_labels):
        '''
        Generates a random audio with its random label
        '''
        x  = {"input_features": generate_random_audio(extractor),
                "labels": generate_random_label(num_labels)}
        return x

    def __getitem__(self, i):
        # if generating on the fly, we generate the sample here. Else, we retrive from list
        if self.onfly == 'y':
            torch.manual_seed(i) # to get the same data for the same index
            return self.generate_random_sample(self.whisper_extractor, self.num_labels)
        else:
            return self.data[i % self.n]
        
    def __len__(self):
        return self.n * self.repeat


# return synthetic data obj
def load_data(conf : config.Config) -> torch.utils.data.Dataset:
    """Simple function to load a dataset based on the provided config object.
    """
    return SyntheticDataWhisper(conf=conf)