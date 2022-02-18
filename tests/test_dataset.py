from typing import Any, Callable, List, Tuple
from pydantic import validator

from config4ml.data.dataset._dataset import DatasetConfig
from torch.utils.data import Dataset, random_split


class CustomDataset(Dataset):
    def __init__(self, root: str, transforms: List[Callable], extra_arg: Any) -> None:
        super().__init__()

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def __len__(self):
        return 1000


class CustomDatasetConfig(DatasetConfig):
    type = "custom_dataset"
    extra_arg: Any

    @validator("extra_arg")
    def validate_extra_arg(cls, v):
        # Validate extra_arg value "v"
        return v

    def build_datasets(self) -> Tuple[Dataset, Dataset]:

        assert self.split == "default", "In this demo only 'default' split is supported"

        dset = CustomDataset(
            root=self.root,
            transforms=self.transform_callables,
            extra_arg=self.extra_arg,
        )

        num_samples_train = int(0.7 * len(dset))
        num_samples_val = len(dset) - num_samples_train
        dset_train, dset_val = random_split(dset, [num_samples_train, num_samples_val])
        return dset_train, dset_val


class TestClass:
    def test_dataset_config(self):

        cfg_obj = {
            "type": "custom_dataset",
            "root": "PATH/TO/DSET/ROOT",
            "split": "default",
            "extra_arg": 32.4,
            "dataloader": {"batch_size": 32},
        }

        cfg = CustomDatasetConfig.parse_obj(cfg_obj)

        dset_train, dset_val = cfg.build_datasets()

        assert isinstance(dset_train, Dataset)
        assert isinstance(dset_val, Dataset)
