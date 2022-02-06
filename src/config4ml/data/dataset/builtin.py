__all__ = ["MNISTConfig"]
from typing import Tuple

from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from ..utils import chain_transforms
from . import DatasetConfig


class MNISTConfig(DatasetConfig):
    type = "mnist"
    download: bool = True

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dset = MNIST(
            self.root,
            train=True,
            transform=chain_transforms(self.transform_callables),
            download=self.download,
        )
        test_dset = MNIST(
            self.root,
            train=False,
            transform=chain_transforms(self.transform_callables),
            download=self.download,
        )
        return train_dset, test_dset
