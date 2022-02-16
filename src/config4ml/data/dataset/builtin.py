"""
Builtin templates for well known benchmark datasets
"""
from typing import Tuple

from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from ..utils import chain_transforms
from . import DatasetConfig


class MNISTConfig(DatasetConfig):
    """Template for configuration class for the MNIST dataset

    Based on torchvision's MNIST implementation

    Attributes:
        type (str): Always equals to "mnist"
        download (bool): Dataset download flag
    """

    type = "mnist"
    download: bool = True

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Builds and returns train/val datasets

        Returns:
            Dataset: Training dataset (subclass of torch.utils.data.DataSet)
            Dataset: Validation dataset (subclass of torch.utils.data.DataSet)
        """
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
