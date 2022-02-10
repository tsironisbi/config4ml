from config4ml.lightning import BaseLoggerConfig, TrainerConfig, select_logger
from config4ml.lightning.extra import ConsoleLogger
from pydantic import BaseModel, validator
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger


class TestClass:
    def test_loggers(self):
        class DemoConfig(BaseModel):
            logger: BaseLoggerConfig

            @validator("logger", pre=True)
            def v_logger(cls, v: dict) -> BaseLoggerConfig:
                return select_logger(v)

        cfg_neptune = {
            "logger": {
                "type": "neptune",
                "api_key": "xxxxxx",
                "project": "ssssss",
                "tags": ["A", "B", "C"],
            }
        }
        cfg1 = DemoConfig.parse_obj(cfg_neptune)
        assert isinstance(cfg1.logger.logger, NeptuneLogger)

        cfg_tensorboard = {"logger": {"type": "tensorboard", "save_dir": "./tb"}}
        cfg2 = DemoConfig.parse_obj(cfg_tensorboard)
        assert isinstance(cfg2.logger.logger, TensorBoardLogger)

        cfg_console = {"logger": {"type": "console"}}
        cfg3 = DemoConfig.parse_obj(cfg_console)
        assert isinstance(cfg3.logger.logger, ConsoleLogger)

    def test_trainer_config(self):

        cfg_obj = {
            "accelerator": "cpu",
            "max_epochs": 50,
            "callbacks": [
                {
                    "type": "model_checkpoint",
                    "monitor": "accuracy",
                    "save_top_k": 3,
                    "save_last": True,
                },
                {"type": "early_stopping", "monitor": "accuracy"},
            ],
            "logger": {
                "type": "neptune",
                "api_key": "xxxxxx",
                "project": "ssssss",
                "tags": ["A", "B", "C"],
            },
        }

        cfg = TrainerConfig.parse_obj(cfg_obj)

        trainer = cfg.trainer

        assert isinstance(trainer, Trainer)
