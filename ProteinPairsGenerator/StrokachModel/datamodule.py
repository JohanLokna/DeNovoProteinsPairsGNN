# General imports
from pathlib import Path
from typing import Union, List

# Pytorch imports
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

# Local imports
from .loader import StrokachLoader


class StrokachDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset : Dataset,
        trainSet: Union[Path, List[Path]],
        valSet: Union[Path, List[Path]],
        testSet: Union[Path, List[Path]],
        teacher: Union[str, None] = None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.trainSet = trainSet
        self.valSet = valSet
        self.testSet = testSet
        self.teacher = teacher

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.trainSet), teacher=self.teacher)

    def val_dataloader(self):
        return StrokachLoader(
            self.dataset.getSubset(self.valSet), 
            teacher=self.teacher,
            num_workers = 8,
            prefetch_factor = 100            
        )

    def test_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.testSet), teacher=self.teacher)

    def transfer_batch_to_device(self, x, device):
        x.__dict__.update((k, v.to(device=device)) for k, v in x.__dict__.items() if isinstance(v, torch.Tensor))
        return x
