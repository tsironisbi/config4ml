from abc import abstractmethod
from typing import Union
from pydantic import BaseModel, validator
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
from torch.optim import Optimizer


class InvalidLRScheduler(Exception):
    pass


class BaseLRSchedulerConfig(BaseModel):
    type: str = "abstract"

    @abstractmethod
    def get_instance(
        self, optimizer: Optimizer
    ) -> Union[_LRScheduler, ReduceLROnPlateau]:
        pass


class StepLRSchedulerConfig(BaseLRSchedulerConfig):
    type = "stepLR"
    step_size: int
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False

    def get_instance(self, optimizer: Optimizer) -> _LRScheduler:
        assert isinstance(optimizer, Optimizer)
        kwargs = self.dict()
        del kwargs["type"]
        return StepLR(optimizer, **kwargs)


class ReduceOnPlateauLRSchedulerConfig(BaseLRSchedulerConfig):
    type = "reduce_on_plateau_LR"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 0.0001
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-08
    verbose: bool = False

    def get_instance(self, optimizer: Optimizer) -> ReduceLROnPlateau:
        assert isinstance(optimizer, Optimizer)
        kwargs = self.dict()
        del kwargs["type"]
        return ReduceLROnPlateau(optimizer, **kwargs)


class ExponentialLRSchedulerConfig(BaseLRSchedulerConfig):
    type = "exponentialLR"
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False

    def get_instance(self, optimizer: Optimizer) -> _LRScheduler:
        assert isinstance(optimizer, Optimizer)
        kwargs = self.dict()
        del kwargs["type"]
        return ExponentialLR(optimizer, **kwargs)


def select_lr_scheduler(v: dict) -> BaseLRSchedulerConfig:
    if v["type"] == "stepLR":
        return StepLRSchedulerConfig.parse_obj(v)
    elif v["type"] == "reduce_on_plateau_LR":
        return ReduceOnPlateauLRSchedulerConfig.parse_obj(v)
    elif v["type"] == "exponentialLR":
        return ExponentialLRSchedulerConfig.parse_obj(v)
    else:
        raise InvalidLRScheduler(f"{v['type']} is not a valid lr scheduler type")
