# General imports
from pathlib import Path
from typing import Union, List

# Pytorch imports
from torch.utils.data import Dataset
import pytorch_lightning as pl

# Local imports
from ProteinPairsGenerator.IngrahamModel import IngrahamLoader

class IngrahamDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset : Dataset,
        trainSet: Union[Path, List[Path]],
        valSet: Union[Path, List[Path]],
        testSet: Union[Path, List[Path]],
        batchSize: int = 1
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.trainSet = trainSet
        self.valSet = valSet
        self.testSet = testSet
        self.batchSize = batchSize

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return IngrahamLoader(self.dataset.getSubset(self.trainSet), batch_size=self.batchSize)

    def val_dataloader(self):
        return IngrahamLoader(self.dataset.getSubset(self.valSet), batch_size=self.batchSize)

    def test_dataloader(self):
        return IngrahamLoader(self.dataset.getSubset(self.testSet), batch_size=self.batchSize)

    def transfer_batch_to_device(self, x, device):
        (X, S, l, v), y, mask = x
        return (X.to(device=device), S.to(device=device), l, v.to(device=device)), y.to(device=device), mask.to(device=device)
