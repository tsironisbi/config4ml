from typing import List, Optional
from matplotlib.pyplot import cla

from pydantic import BaseModel, validator
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)

from .extra import ConsoleLogger


class InvalidLogger(Exception):
    pass


class BaseLoggerConfig(BaseModel):
    type: str = "abstract"

    @property
    def logger(self) -> LightningLoggerBase:
        raise NotImplementedError


class NeptuneLoggerConfig(BaseLoggerConfig):
    type = "neptune"
    api_key: str
    project: str
    tags: List[str]

    @property
    def logger(self) -> LightningLoggerBase:
        kwargs = self.dict()
        kwargs.pop("type")
        return NeptuneLogger(**kwargs)


class TensorboardLoggerConfig(BaseLoggerConfig):
    type = "tensorboard"
    save_dir: str
    name: Optional[str] = None

    @property
    def logger(self) -> LightningLoggerBase:
        kwargs = self.dict()
        kwargs.pop("type")
        return TensorBoardLogger(**kwargs)


class WandLoggerConfig(BaseLoggerConfig):
    type = "wandb"
    project: Optional[str] = None
    experiment: Optional[str] = None
    name: Optional[str] = None
    save_dir: Optional[str] = None
    offline: bool = False
    id: Optional[int] = None
    anonymous: Optional[bool] = None
    version: Optional[int] = None
    log_model: bool = False
    prefix: str = ""

    @property
    def logger(self) -> LightningLoggerBase:
        kwargs = self.dict()
        kwargs.pop("type")
        return WandbLogger(**kwargs)


class ConsoleLoggerConfig(BaseLoggerConfig):
    type = "console"

    @property
    def logger(self) -> LightningLoggerBase:
        return ConsoleLogger()


def select_logger(v: dict) -> BaseLoggerConfig:
    if v["type"] == "neptune":
        return NeptuneLoggerConfig.parse_obj(v)
    elif v["type"] == "tensorboard":
        return TensorboardLoggerConfig.parse_obj(v)
    elif v["type"] == "wandb":
        return WandLoggerConfig.parse_obj(v)
    elif v["type"] == "console":
        return ConsoleLoggerConfig.parse_obj(v)
    else:
        raise InvalidLogger(f"{v['type']} is not a valid logger type")
