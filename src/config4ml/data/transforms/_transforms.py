from typing import Any, Callable, Dict, List, Optional, OrderedDict, TypeVar

from pydantic import BaseModel, root_validator, validator


class InvalidTransformError(Exception):
    pass


class Registry:
    _transforms: OrderedDict[str, Callable] = OrderedDict()

    @classmethod
    def register(cls, T: Callable, key: Optional[str] = None) -> None:

        assert isinstance(T, Callable), "Transform to be registered must be a Callable"

        if key == None:
            key = T.__name__

        cls._transforms[key] = T

    @classmethod
    def get_transform(cls, key: str) -> Callable:
        if key not in cls._transforms:
            raise InvalidTransformError(f"{key=} is not a valid transform name")
        return cls._transforms[key]


class TransformConfig(BaseModel):
    fun: Callable
    kwargs: Dict[str, Any] = dict()

    @root_validator(pre=True)
    def select_transform(cls, values: Dict):
        if isinstance(values["fun"], Callable):
            return values

        assert isinstance(values["fun"], str)
        kwargs = values["kwargs"] if "kwargs" in values else {}
        transform_fun = Registry.get_transform(values["fun"])

        values["fun"] = transform_fun(**kwargs)
        return values


if __name__ == "__main__":

    def T(v):
        def apply(x):
            return x + v

        return apply

    Registry.register(T)
    print(Registry._transforms)
