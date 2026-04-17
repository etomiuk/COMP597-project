# COMP597 project
This project measures the energy and compute resource consumption of OpenAI's Whisper model training. Details on the project, starter code, and initial setup instructions can be found [here](https://github.com/OMichaud0/COMP597-starter-code).

## Whisper model & dataset
| Component    | Path | Description | 
| -------- | ------- | ------- |
| Model  | `src/config/data/whisper_data/config.py`    | `WhisperForAudioClassification` is used for this project and is trained using the provided `SimpleTrainer` |
| Dataset | `src/data/whisper_data/data.py`     | Both dataset generation modes are included in this file. The mode is set using the command line argument `--data_configs.whisper_data.onfly` set to `y` for data generation on the fly and `n` for data generation before training. |

### Model parameters
To train the Whisper model, use the following command line arguments with these values:
| Parameter    | Argument | Value |
| -------- | ------- | ------- |
| Model | `--model` | `whisper` |
| Dataset | `--data` | `whisper_data` | 
| Experiment | `--trainer_stats` | See [Experiments](#experiments) section below |

### Additional parameters
Additional model configurations can be set using the following command line arguments:

| Parameter    | Argument |
| -------- | ------- |
| Number of labels | `--data_configs.whisper_data.num_labels` |
| Number of unique training samples | `--data_configs.whisper_data.num_samples` |
| Number of times to repeat each sample | `--data_configs.whisper_data.repeat` |
| Number of workers | `--data_configs.whisper_data.num_workers` |
| Data creation mode | `--data_configs.whisper_data.onfly` |

They are defined in `src/config/data/whisper_data/config.py`

## Experiments
All data collection is handled in the classes defined in `src/trainer/stats/stats_data.py`. Different experiments create objects of these classes to collect the appropriate data.

Experiments are summarized in the following table, along with their value to the `--trainer_stats` command line argument used to run those experiments. The value of the argument matches the code file that runs the experiment, all found in `src/trainer/stats/`. 

| Experiment | Value | Data directory | Description |
| -------- | ------- | ------- | ------- |
| End-to-end timing | `timing_train` | `final_data_analysis\timing_data\train` | Timing of the entire training loop |
| CodeCarbon timing | `codecarbon_timed_train` | `final_data_analysis\overhead\train` | Timing of the entire training loop with one CodeCarbon measurement. CodeCarbon measurements are not saved. |
| Per phase timing | `timing_*`, where `*` is one of `fwd`, `bkwd`, `optim`, `step` | `final_data_analysis\timing_data\*` | Timing of each phase separately. For `step`, time is measured per two steps to calculate estimates of batch creation for the on-the-fly data creation mode. |
| Resource per step | `resource_usage_step` | `final_data_analysis\resource_data\step` | Measurements of CPU/GPU utilization and GPU memory every step. |
| Resource per phase | `resource_usage_phase` | `final_data_analysis\resource_data\*`, where `*` is one of `forward`, `backward` | Measurements of CPU/GPU utilization and GPU memory every phase. |
| CodeCarbon per step | `codecarbon_timed_step` | `final_data_analysis\energy_data` | Energy, power, and carbon emission measurements every two steps. |

## Data analysis
Data analysis files for the experiments are described in the following table. They are all found under `final_data_analysis\analysis\`
| Description | Analysis directory | Description |
| -------- | ------- | ------- |
| Per phase timing | `time_analysis.ipynb` | Plots for timing per phase and timing of sample generation |
| Sample generation histogram | `sample_generation_analysis.ipynb` | Sample generation time historgram |
| Overhead | `overhead_all_exp.ipynb` | Plots for overhead of each experiment |
| Compute resources | `resource_analysis_step.ipynb` and `resource_analysis_per_phase.ipynb` | Plots for resource consumption. |
| CodeCarbon | `energy_analysis_train.ipynb`, `energy_analysis_train_avg.ipynb`, and `energy_analysis_per_step.ipynb` | Plots for energy, power, and carbon for the whole training and per step. The `energy_analysis_train_avg` calculates the average over the whole training by taking the per step files and averaging them out. This was used to generate the GPU power plot specifically.|
