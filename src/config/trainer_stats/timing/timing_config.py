from src.config.util.base_config import _Arg, _BaseConfig

config_name="timing"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_step = _Arg(type=bool, help="Whether to track the step time", default=False)
        self._arg_fwd = _Arg(type=bool, help="Whether to track the forward pass time", default=False)
        self._arg_bkwd = _Arg(type=bool, help="Whether to track the backward pass time", default=False)
        self._arg_optim = _Arg(type=bool, help="Whether to track the optimization time", default=False)
