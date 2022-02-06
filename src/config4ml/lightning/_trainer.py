from typing import List, Optional, Union

from pydantic import BaseModel, validator
from pytorch_lightning import Trainer

from . import CallbackConfig, select_callback


class TrainerConfig(BaseModel):
    accelerator: str = "gpu"
    accumulate_grad_batches: Optional[int] = None
    max_epochs: int = 200
    check_val_every_n_epoch: int = 5
    callbacks: List[CallbackConfig] = []

    @validator("accelerator")
    def validate_accelerator(cls, v):
        assert v in ["cpu", "gpu", "tpu", "ipu"], f"{v} is an invalid accelerator type"
        return v

    @validator("callbacks", pre=True, each_item=True)
    def validate_callback(cls, v: Union[CallbackConfig, dict]):
        if isinstance(v, CallbackConfig):
            return v

        assert isinstance(v, dict)
        return select_callback(v)

    @property
    def trainer(self):

        callbacks_list = [cb.callback for cb in self.callbacks]

        kwargs = self.dict()
        kwargs["callbacks"] = callbacks_list
        trainer_instance = Trainer(**kwargs)

        return trainer_instance
