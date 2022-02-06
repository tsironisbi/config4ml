from typing import Optional

from pydantic import BaseModel, root_validator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class InvalidCallback(Exception):
    pass


class CallbackConfig(BaseModel):
    type: str = "abstract"

    @property
    def callback(self):
        raise NotImplementedError


class ModelCheckpointConfig(CallbackConfig):
    type: str = "model_checkpoint"
    monitor: str
    filename: Optional[str] = None
    save_on_train_epoch_end: bool = False
    save_top_k: int = 3
    save_last: bool = True
    mode: str = "max"

    @root_validator
    def gen_filename(cls, values):
        if values["filename"] is None:
            values["filename"] = f'{{epoch}}-{{{values["monitor"]}:.2f}}'

        return values

    @property
    def callback(self):
        d = self.dict()
        d.pop("type")
        return ModelCheckpoint(**d)


class EarlyStoppingConfig(CallbackConfig):
    type: str = "early_stopping"
    monitor: str

    @property
    def callback(self):
        d = self.dict()
        d.pop("type")
        return EarlyStopping(**d)


def select_callback(v: dict) -> CallbackConfig:
    if v["type"] == "model_checkpoint":
        return ModelCheckpointConfig.parse_obj(v)
    if v["type"] == "early_stopping":
        return EarlyStoppingConfig.parse_obj(v)
    else:
        raise InvalidCallback(f"{v['type']} is not a valid callback type")
