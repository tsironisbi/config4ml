from abc import abstractmethod
from typing import Iterable, Tuple

from pydantic import BaseModel, validator

from torch.optim import SGD, Adam, Optimizer


class InvalidOptimizer(Exception):
    pass


class BaseOptimizerConfig(BaseModel):
    type: str = "abstract"

    @abstractmethod
    def get_instance(self, parameters: Iterable) -> Optimizer:
        pass


class AdamOptimizerConfig(BaseOptimizerConfig):
    type = "adam"
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False

    def get_instance(self, parameters: Iterable) -> Adam:
        kwargs = self.dict()
        del kwargs["type"]
        return Adam(parameters, **kwargs)


class SGDOptimizerConfig(BaseOptimizerConfig):
    type = "sgd"
    lr: float
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False

    def get_instance(self, parameters: Iterable) -> SGD:
        kwargs = self.dict()
        del kwargs["type"]
        return SGD(parameters, **kwargs)


def select_optimizer(v: dict) -> BaseOptimizerConfig:
    if v["type"] == "adam":
        return AdamOptimizerConfig.parse_obj(v)
    elif v["type"] == "sgd":
        return SGDOptimizerConfig.parse_obj(v)
    else:
        raise InvalidOptimizer(f"{v['type']} is not a valid optimizer type")
