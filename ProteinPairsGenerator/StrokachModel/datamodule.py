# General imports
from pathlib import Path
from typing import Union, List

# Pytorch imports
from torch.utils.data import Dataset
import pytorch_lightning as pl

# Local imports
from .loader import StrokachLoader
from ProteinPairsGenerator.DistilationKnowledge import KDTokenizer


class StrokachDataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset : Dataset,
        trainSet: Union[Path, List[Path]],
        valSet: Union[Path, List[Path]],
        testSet: Union[Path, List[Path]],
        kdRegularizer : Union[str, None] = None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.trainSet = trainSet
        self.valSet = valSet
        self.testSet = testSet
        self.tokenizer = {None: None, "TAPE": KDTokenizer()}

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.trainSet), tokenizer=self.tokenizer)

    def val_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.valSet), tokenizer=self.tokenizer)

    def test_dataloader(self):
        return StrokachLoader(self.dataset.getSubset(self.testSet), tokenizer=self.tokenizer)

    def transfer_batch_to_device(self, x, device):
        x.__dict__.update((k, v.to(device=device)) for k, v in x.__dict__.items())
        return x
