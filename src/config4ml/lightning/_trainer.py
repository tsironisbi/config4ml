from typing import List, Optional, Union

from pydantic import BaseModel, root_validator, validator
from pytorch_lightning import Trainer

from . import CallbackConfig, select_callback, BaseLoggerConfig, select_logger
from .extra import ConsoleLogger


class TrainerConfig(BaseModel):
    accelerator: str = "gpu"
    accumulate_grad_batches: Optional[int] = None
    max_epochs: int = 200
    check_val_every_n_epoch: int = 5
    callbacks: List[CallbackConfig] = []
    logger: BaseLoggerConfig
    devices: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

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

    @validator("logger", pre=True)
    def validate_logger(cls, v):
        if isinstance(v, BaseLoggerConfig):
            return v

        assert isinstance(v, dict)
        return select_logger(v)

    @root_validator
    def validate_devices(cls, values):
        if values["accelerator"] != "cpu":
            if values.get("devices", None) is None:
                values["devices"] = 1
        else:
            assert (
                values.get("devices", None) is None
            ), "When accelerator is 'cpu' devices must be None"
        return values

    @property
    def trainer(self):
        print(
            'WARNING! "trainer" property to be deprecated soon. Use .get_trainer_instance method instead'
        )
        return self.get_trainer_instance()

    def get_trainer_instance(self, console_logging_override=False) -> Trainer:
        callbacks_list = [cb.callback for cb in self.callbacks]

        if console_logging_override:
            logger = ConsoleLogger()
        else:
            logger = self.logger.logger

        kwargs = self.dict()
        kwargs["callbacks"] = callbacks_list
        kwargs["logger"] = logger
        trainer_instance = Trainer(**kwargs)

        return trainer_instance
