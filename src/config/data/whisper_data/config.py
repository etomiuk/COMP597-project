from src.config.util.base_config import _Arg, _BaseConfig

class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_num_samples = _Arg(type=int, help="Number of samples to generate for the synthetic dataset.", default=100)