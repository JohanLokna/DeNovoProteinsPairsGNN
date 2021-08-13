from pathlib import Path
import json
import random
from typing import List, Optional

import torch
import pytorch_lightning as pl

from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer
from ProteinPairsGenerator.Data import GeneralData
from ProteinPairsGenerator.utils import seq_to_tensor, AMINO_ACIDS_MAP

class GuessDataset(torch.utils.data.IterableDataset):

    def __init__(self, root : str, xToken : str = "seq", yToken : str = "guess", maskToken = "mask") -> None:
        super().__init__()
        self.tokenizer = AdaptedTAPETokenizer()
        self.xToken = xToken
        self.yToken = yToken
        self.maskToken = maskToken

        self.data = []
        for l in open(root, "r").readlines():

            jsonDict = json.loads(l)

            self.data.append(GeneralData(
              x = seq_to_tensor(jsonDict[self.xToken], AMINO_ACIDS_MAP),
              y = seq_to_tensor(jsonDict[self.yToken], AMINO_ACIDS_MAP),
              mask = torch.BoolTensor(jsonDict[self.maskToken])
            ))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class GuessLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True):

        if shuffle:
            random.shuffle(dataset.data)

        super().__init__(dataset, batch_size, collate_fn=lambda x: x[0])


class GuessDataModule(pl.LightningDataModule):

    def __init__(
        self,
        trainSet: Optional[str] = None,
        valSet: Optional[str] = None,
        testSet: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.trainDataset = GuessDataset(trainSet) if trainSet else None
        self.valDataset = GuessDataset(valSet) if valSet else None
        self.testDataset = GuessDataset(testSet) if testSet else None

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return GuessLoader(
            dataset=self.trainDataset,
            shuffle=True,
        )

    def val_dataloader(self):
        return GuessLoader(
            dataset=self.valDataset
        )

    def test_dataloader(self):
        return GuessLoader(
            dataset=self.testDataset
        )

    def transfer_batch_to_device(self, x, device):
        x.__dict__.update((k, v.to(device=device)) for k, v in x.__dict__.items() if isinstance(v, torch.Tensor))
        return x
