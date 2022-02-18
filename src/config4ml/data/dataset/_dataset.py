"""
Dataset related configuration objects
"""
from abc import abstractmethod
from pydantic import BaseModel
from typing import Callable, Optional, Tuple, List, Union
from torch.utils.data import Dataset, DataLoader

from ..transforms._transforms import TransformConfig


class DataloaderConfig(BaseModel):
    """Configuration class for torch.utils.data.DataLoader objects

    Attributes:
        num_workers (int): Number of worker processes to be spawned by the dataloader
        prefetch_factor (int): How many batches to prefetch
        batch_size (int): Batch size for data-loading
    """

    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    batch_size: int

    def get_train_dataloader(self, dset: Dataset, **extra_kwargs) -> DataLoader:
        """Generates a "training" dataloader from dataloader config options.

        The returned dataloader has it's shuffle property
        to *True* unless overriden from **extra_kwargs.
        The usere may specify any extra kwargs compatible with
        torch.utils.data.DataLoader that *will* override
        the options of the config object

        Args:
            dset (Dataset): The dataset from which to load data

        Returns:
            DataLoader: Dataloader to use for training
        """
        kwargs = self.dict()
        kwargs["shuffle"] = True
        kwargs.update(extra_kwargs)

        return DataLoader(dset, **kwargs)

    def get_val_dataloader(self, dset: Dataset, **extra_kwargs) -> DataLoader:
        """Generates a "validation" dataloader from dataloader config options.

        The returned dataloader has it's shuffle property
        to *False* unless overriden from **extra_kwargs.
        The usere may specify any extra kwargs compatible with
        torch.utils.data.DataLoader that *will* override
        the options of the config object

        Args:
            dset (Dataset): The dataset from which to load data

        Returns:
            DataLoader: Dataloader to use for validation
        """
        kwargs = self.dict()
        kwargs["shuffle"] = False
        kwargs.update(extra_kwargs)

        return DataLoader(dset, **kwargs)


class DatasetConfig(BaseModel):
    """Configuration class for torch.utils.data.DataSet compatible objects

    Attributes:
        type (str): Dataset type (unique identifier)
        root (str): Path to dataset root
        split (str | float): How to split the dataset into train / val subsets
        transforms (list): List of transforms (as TransformConfig) to be performed
            when loading data
        dataloader (DataloaderConfig): Configuration for the dataloaders
    """

    type: str = "abstract_dataset"
    root: str
    split: Union[str, float] = "default"
    transforms: List[TransformConfig] = []
    dataloader: DataloaderConfig = DataloaderConfig()

    @property
    def transform_callables(self) -> List[Callable]:
        """Generate a list of callable transforms from their configurations

        Returns:
            List[Callable]: List of callable transforms
        """
        return [T.fun for T in self.transforms]

    @abstractmethod
    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Abstract method for building train/val datasets

        Returns:
            Dataset: Training dataset (subclass of torch.utils.data.DataSet)
            Dataset: Validation dataset (subclass of torch.utils.data.DataSet)
        """
        pass

    def build_dataloaders(self, **extra_kwargs) -> Tuple[DataLoader, DataLoader]:
        """Builds and returns train and validation dataloaders

        Returns:
            DataLoader: Training dataloader
            DataLoader: Validation dataloader
        """
        dset_train, dset_val = self.build_datasets()
        dloader_train = self.dataloader.get_train_dataloader(dset_train, **extra_kwargs)
        dloader_val = self.dataloader.get_val_dataloader(dset_val, **extra_kwargs)
        return dloader_train, dloader_val
