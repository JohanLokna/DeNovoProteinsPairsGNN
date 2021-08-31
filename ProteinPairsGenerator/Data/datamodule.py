# General imports
from pathlib import Path
from typing import Union, List, Optional, Callable

# Pytorch imports
import torch
import pytorch_lightning as pl

# Local imports
from .proteinNetDataset import ProteinNetDataset

"""
    Base class datamodule which is specialized for the different models
"""
class BERTDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root : Union[Path, str],
        loaderClass,
        trainSet: Optional[Union[Path, List[Path]]] = None,
        valSet: Optional[List[Path]] = None,
        testSet: Optional[List[Path]] = None,
        teacher: Optional[str] = None,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        batch_size: int = 1,
        batch_builder: Optional[Callable] = None
    ) -> None:
        super().__init__()

        # Ensure that root is of type Path
        if isinstance(root, str):
            root = Path(root)

        # Set up datsets
        self.trainSet = trainSet if not trainSet is None else [root.joinpath("processed/training_100")]
        self.valSet = valSet if not valSet is None else [root.joinpath("processed/validation")]
        self.testSet = testSet if not testSet is None else [root.joinpath("processed/testing")]
        self.dataset = ProteinNetDataset(
            root = root,
            subsets = self.trainSet + self.valSet + self.testSet
        )

        # Set up members
        self.teacher = teacher
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.loaderClass = loaderClass

        # Ensure that either batch size 1 is used or  
        if batch_size != 1 and not (batch_builder is None):
            raise RuntimeError("Bad batching")

        # Set up batches
        indeciesList = [self.dataset.getSubset(self.trainSet), 
                        self.dataset.getSubset(self.valSet), 
                        self.dataset.getSubset(self.testSet)]

        if batch_builder is None:
            self.train_indices, self.val_indices, self.test_indices = indeciesList
        else:
            self.train_indices, self.val_indices, self.test_indices = [batch_builder(indices) for indices in indeciesList]

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.loaderClass(
            dataset=self.train_indices,
            teacher=self.teacher,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        return self.loaderClass(dataset=self.val_indices, teacher=self.teacher)

    def test_dataloader(self):
        return self.loaderClass(dataset=self.test_indices, teacher=self.teacher)

    def transfer_batch_to_device(self, x, device):
        x.__dict__.update((k, v.to(device=device)) for k, v in x.__dict__.items() if isinstance(v, torch.Tensor))
        return x
