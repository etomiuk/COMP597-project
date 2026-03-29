# === import necessary modules ===
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class
import src.trainer.stats as trainer_stats # Trainer statistics module

# === import necessary external modules ===
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers


def init_whisper_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """
    Initializes the optimizer for the Whisper model.
    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The Whisper model.
    Returns:
        optim.Optimizer: The initialized optimizer.

    Note: it's the same as GPT example
    """
    # This is a simple AdamW optimizer with weight decay. Choose different optimizers as needed.
    # Note: The learning rate is taken from the configuration object. Adjust it as needed for different models and training setups based on the loss function.
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)

def pre_init_whisper(conf: config.Config, dataset: data.Dataset) -> Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]:
    """
    Obtains the Whisper model: WhisperForAudioClassification. Nothing is done to the dataset
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]: The Whisper model, dataset, tokenizer and data collator.
    """
    model = transformers.WhisperForAudioClassification.from_pretrained("openai/whisper-tiny", num_labels=conf.data_configs.whisper_data.num_labels) 

    return model, dataset

################################################################################
#################################    Simple    #################################
################################################################################

# change the simple trainer modify the process_batch function
def simple_trainer(conf : config.Config, model : transformers.WhisperModel, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for Whisper model. Uses the SimpleTrainer from src/trainer/simple.py.
    Args:
        conf (config.Config): The configuration object.
        model (transformers.WhisperModel): The Whisper model to train.
        dataset (data.Dataset): The dataset to train on.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        data_collator (transformers.DataCollatorForLanguageModeling): The data collator to use.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The simple trainer and a dictionary with additional options.
    """
    loader = data.DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.data_configs.whisper_data.num_workers) # DataLoader for batching the dataset
    model = model.cuda() # Move the model to GPU
    optimizer = init_whisper_optim(conf, model) # Get the optimizer
    scheduler = transformers.get_scheduler( # Linear learning rate decay scheduler
        "linear", 
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader), 
    )

    # Return the SimpleTrainer with the initialized components
    # class MyTrainer inherits from simple.Simpletrainer with process-batch function + init
    return trainer.SimpleTrainer(loader=loader, model=model, optimizer=optimizer, lr_scheduler=scheduler, device=model.device, stats=trainer_stats.init_from_conf(conf=conf, device=model.device, num_train_steps=len(loader))), None

################################################################################
##################################    Init    ##################################
################################################################################

def whisper_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the Whisper model and returns the appropriate trainer based on the configuration.
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The initialized trainer and a dictionary with additional options.
    """
    model, dataset = pre_init_whisper(conf, dataset) # get the Whisper model
    if conf.trainer == "simple": 
        return simple_trainer(conf, model, dataset)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")




