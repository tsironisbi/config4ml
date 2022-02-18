from config4ml.data.dataset.builtin import MNISTConfig
from torch.utils.data import Dataset


class TestClass:
    def test_dataset_config(self):

        cfg_obj = {
            "type": "mnist",
            "root": "./mnist",
            "split": "default",
            "download": True,
            "dataloader": {"batch_size": 32},
        }

        cfg = MNISTConfig.parse_obj(cfg_obj)

        dset_train, dset_val = cfg.build_datasets()

        assert isinstance(dset_train, Dataset)
        assert isinstance(dset_val, Dataset)
