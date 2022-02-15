from typing import Callable
import config4ml.data.transforms as transforms


def T() -> Callable:
    def apply(x: int) -> int:
        return x

    return apply


class TestClass:
    def test_register(self):
        registry = transforms.Registry
        registry.register(T)

        assert T.__name__ in registry._transforms

    def test_config_generation(self):
        registry = transforms.Registry
        registry.register(T)

        cfg_dict = {"fun": T.__name__, "kwargs": {"value": 5}}
        cfg = transforms.TransformConfig.parse_obj(cfg_dict)

        assert isinstance(cfg.fun, Callable)
        assert cfg.kwargs["value"] == 5
        assert cfg.fun(2) == 7
