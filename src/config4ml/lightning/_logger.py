from typing import List, Optional

from pydantic import BaseModel, validator
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    NeptuneLogger,
    TensorBoardLogger,
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


class ConsoleLoggerConfig(BaseLoggerConfig):
    type = "console"

    @property
    def logger(self) -> LightningLoggerBase:
        return ConsoleLogger()


def select_logger(v: dict) -> BaseLoggerConfig:
    if v["type"] == "neptune":
        return NeptuneLoggerConfig.parse_obj(v)
    if v["type"] == "tensorboard":
        return TensorboardLoggerConfig.parse_obj(v)
    if v["type"] == "console":
        return ConsoleLoggerConfig.parse_obj(v)
    else:
        raise InvalidLogger(f"{v['type']} is not a valid logger type")
