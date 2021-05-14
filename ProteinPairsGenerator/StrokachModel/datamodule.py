# General imports
from pathlib import Path
from typing import Union, List

# Pytorch imports
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

# Local imports
from .loader import StrokachLoader
from ProteinPairsGenerator.Data import ProteinNetDataset


class StrokachDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root : Union[Path, str],
        trainSet: Union[Path, List[Path], None] = None,
        valSet: Union[List[Path], None] = None,
        testSet: Union[List[Path], None] = None,
        teacher: Union[str, None] = None
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        self.trainSet = trainSet if not trainSet is None else [root.joinpath("processed/training_100")]
        self.valSet = valSet if not valSet is None else [root.joinpath("processed/validation")]
        self.testSet = testSet if not testSet is None else [root.joinpath("processed/testing")]
        self.dataset = ProteinNetDataset(
            root = root,
            subsets = self.trainSet + self.valSet + self.trainSet,
            batchSize = 11000,
        )
        self.teacher = teacher

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return StrokachLoader(
            self.dataset.getSubset(self.trainSet), 
            teacher=self.teacher,
            num_workers=1,
            pin_memory=True
        )

    def val_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.valSet), teacher=self.teacher)

    def test_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.testSet), teacher=self.teacher)

    def transfer_batch_to_device(self, x, device):
        x.__dict__.update((k, v.to(device=device)) for k, v in x.__dict__.items() if isinstance(v, torch.Tensor))
        return x
