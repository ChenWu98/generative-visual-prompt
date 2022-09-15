# Created by Chen Henry Wu
import numpy as np
import torch
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset
from datasets import DatasetDict

MAX_VALUE_POSE = (1/9) * np.pi


class Preprocessor(object):

    def __init__(self, args):
        self.args = args

    def preprocess(self, raw_datasets: DatasetDict, cache_root: str):
        assert len(raw_datasets) == 3  # Not always.
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return {
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset,
        }


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        n_sample = 1024

        self.uniform_xy = Uniform(
            low=torch.FloatTensor([-MAX_VALUE_POSE, -MAX_VALUE_POSE]),
            high=torch.FloatTensor([MAX_VALUE_POSE, MAX_VALUE_POSE]),
        )
        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "model_kwargs": ["sample_id", "alpha", ]
            }
            for idx in range(n_sample)
        ]

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items()}

        # Add alpha.
        xy = self.uniform_xy.sample()
        z = torch.zeros_like(xy[:1])
        data["alpha"] = torch.cat([xy, z], dim=0)
        data["model_kwargs"] = data["model_kwargs"] + ["alpha", ]

        return data

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        sqrt_n_sample = 8
        n_sample = sqrt_n_sample ** 2
        grids = np.linspace(-MAX_VALUE_POSE, MAX_VALUE_POSE, sqrt_n_sample)

        self.xy = [
            torch.FloatTensor([_x, _y])
            for _x in grids for _y in grids
        ]
        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "model_kwargs": ["sample_id", "alpha", ]
            }
            for idx in range(n_sample)
        ]

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items()}

        # Add alpha.
        xy = self.xy[index]
        z = torch.zeros_like(xy[:1])
        data["alpha"] = torch.cat([xy, z], dim=0)
        data["model_kwargs"] = data["model_kwargs"] + ["alpha", ]

        return data

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        sqrt_n_sample = 8
        n_sample = sqrt_n_sample ** 2
        grids = np.linspace(-MAX_VALUE_POSE, MAX_VALUE_POSE, sqrt_n_sample)

        self.xy = [
            torch.FloatTensor([_x, _y])
            for _x in grids for _y in grids
        ]
        self.data = [
            {
                "sample_id": torch.LongTensor([idx]).squeeze(0),
                "model_kwargs": ["sample_id", "alpha", ]
            }
            for idx in range(n_sample)
        ]

    def __getitem__(self, index):
        data = {k: v for k, v in self.data[index].items()}

        # Add alpha.
        xy = self.xy[index]
        z = torch.zeros_like(xy[:1])
        data["alpha"] = torch.cat([xy, z], dim=0)
        data["model_kwargs"] = data["model_kwargs"] + ["alpha", ]

        return data

    def __len__(self):
        return len(self.data)
