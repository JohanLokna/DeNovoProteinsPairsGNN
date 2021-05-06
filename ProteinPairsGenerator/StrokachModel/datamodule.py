# General imports
from pathlib import Path
from typing import Union, List

# Pytorch imports
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
        testSet: Union[Path, List[Path]]
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.trainSet = trainSet
        self.valSet = valSet
        self.testSet = testSet

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.trainSet))

    def val_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.valSet))

    def test_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.testSet))

    def transfer_batch_to_device(self, batch, device):
        for x in batch:
            x.__dict__.update((k, v.to(device=device)) for k, v in x.__dict__.items())
        return batch
