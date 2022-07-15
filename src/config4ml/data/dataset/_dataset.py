from abc import abstractmethod
from pydantic import BaseModel, validator, root_validator
from typing import Callable, Optional, Tuple, List, Dict, Any, Type, Union
from torch.utils.data import Dataset, DataLoader

from ..transforms._transforms import TransformConfig


class DataloaderConfig(BaseModel):
    num_workers: int = 0
    prefetch_factor: int = 1
    batch_size: int = 32

    def get_train_dataloader(self, dset: Dataset, **extra_kwargs) -> DataLoader:

        kwargs = self.dict()
        kwargs["shuffle"] = True
        kwargs.update(extra_kwargs)

        return DataLoader(dset, **kwargs)

    def get_val_dataloader(self, dset: Dataset, **extra_kwargs) -> DataLoader:

        kwargs = self.dict()
        kwargs["shuffle"] = False
        kwargs.update(extra_kwargs)

        return DataLoader(dset, **kwargs)


class DatasetConfig(BaseModel):
    type: str = "abstract_dataset"
    root: str
    split: Union[str, float] = "default"
    train_transforms: List[TransformConfig] = []
    val_transforms: List[TransformConfig] = []
    dataloader: DataloaderConfig = DataloaderConfig()

    @property
    def transform_callables(self, mode:str) -> List[Callable]:
        assert mode in ["train", "val"]
        if mode == "train":
            return [T.fun for T in self.train_transforms]
        else:
            return [T.fun for T in self.val_transforms]
    
    @abstractmethod
    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        pass

    def build_dataloaders(self, **extra_kwargs) -> Tuple[DataLoader, DataLoader]:
        dset_train, dset_val = self.build_datasets()
        dloader_train = self.dataloader.get_train_dataloader(dset_train, **extra_kwargs)
        dloader_val = self.dataloader.get_val_dataloader(dset_val, **extra_kwargs)
        return dloader_train, dloader_val


# def select_dataset_by_type(typ_str: str) -> Type:
#     if typ_str == "ntua":
#         v = NTUADatasetConfig
#     elif typ_str == "breizhcrops":
#         v = BreizhCropsDatasetConfig
#     else:
#         raise ValueError(f"Dataset type {typ_str} is unsupported!")
#     return v


# def dataset_selection_validator(v):
#     if isinstance(v, DatasetConfig):
#         return v

#     assert isinstance(v, dict), "Expected dict for Dataset Config"
#     assert "type" in v, "'type' arg missing for dataset config"

#     dset_cls: Type = select_dataset_by_type(v["type"])
#     new_v: DatasetConfig = dset_cls.parse_obj(v)

#     return new_v
