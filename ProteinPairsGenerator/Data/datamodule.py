# General imports
from pathlib import Path
from typing import Union, List

# Pytorch imports
import torch
import pytorch_lightning as pl

# Local imports
from .proteinNetDataset import ProteinNetDataset


class BERTDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root : Union[Path, str],
        loaderClass,
        trainSet: Union[Path, List[Path], None] = None,
        valSet: Union[List[Path], None] = None,
        testSet: Union[List[Path], None] = None,
        teacher: Union[str, None] = None,
        num_workers: int = 1,
        prefetch_factor: int = 2
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        self.trainSet = trainSet if not trainSet is None else [root.joinpath("processed/training_100")]
        self.valSet = valSet if not valSet is None else [root.joinpath("processed/validation")]
        self.testSet = testSet if not testSet is None else [root.joinpath("processed/testing")]
        self.dataset = ProteinNetDataset(
            root = root,
            subsets = self.trainSet + self.valSet + self.trainSet
        )
        self.teacher = teacher
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.loaderClass = loaderClass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.loaderClass(
            self.dataset.getSubset(self.trainSet), 
            teacher=self.teacher,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return self.loaderClass(self.dataset.getSubset(self.valSet), teacher=self.teacher)

    def test_dataloader(self):
        return self.loaderClass(self.dataset.getSubset(self.testSet), teacher=self.teacher)

    def transfer_batch_to_device(self, x, device):
        x.__dict__.update((k, v.to(device=device)) for k, v in x.__dict__.items() if isinstance(v, torch.Tensor))
        return x
