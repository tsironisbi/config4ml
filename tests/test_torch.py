from config4ml.torch.lr_scheduler import (
    StepLRSchedulerConfig,
    select_lr_scheduler,
    ExponentialLRSchedulerConfig,
    ReduceOnPlateauLRSchedulerConfig,
)
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

model = nn.Linear(10, 10)
optim: Optimizer = Adam(model.parameters(), lr=0.01)


class TestClass:
    def test_lr_schedulers(self):

        cfg_steplr = {"type": "stepLR", "step_size": 5}

        cfg = select_lr_scheduler(cfg_steplr)
        assert isinstance(cfg, StepLRSchedulerConfig)
        assert isinstance(cfg.get_instance(optimizer=optim), _LRScheduler)

        cfg_explr = {"type": "exponentialLR", "gamma": 0.1}

        cfg = select_lr_scheduler(cfg_explr)
        assert isinstance(cfg, ExponentialLRSchedulerConfig)
        assert isinstance(cfg.get_instance(optimizer=optim), _LRScheduler)

        cfg_roplr = {"type": "reduce_on_plateau_LR", "patience": 5, "cooldown": 2}

        cfg = select_lr_scheduler(cfg_roplr)
        assert isinstance(cfg, ReduceOnPlateauLRSchedulerConfig)
        assert isinstance(cfg.get_instance(optimizer=optim), ReduceLROnPlateau)
