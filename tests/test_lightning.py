from config4ml.lightning import TrainerConfig
from pytorch_lightning import Trainer


class TestClass:
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
        }

        cfg = TrainerConfig.parse_obj(cfg_obj)

        trainer = cfg.trainer

        assert isinstance(trainer, Trainer)
