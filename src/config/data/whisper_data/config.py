from src.config.util.base_config import _Arg, _BaseConfig

class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_num_labels = _Arg(type=int, help="Number of labels for the synthetic dataset.", default=10)
        self._arg_num_samples = _Arg(type=int, help="Number of samples for the synthetic dataset.", default=5500)
        self._arg_repeat = _Arg(type=int, help="Number of times to repeat the unique samples in the synthetic dataset.", default=1)
        self._arg_num_workers = _Arg(type=int, help="Number of workers for DataLoader", default=0)
        self._arg_onfly = _Arg(type=str, help="Generate data on the fly vs before the training", default='n')