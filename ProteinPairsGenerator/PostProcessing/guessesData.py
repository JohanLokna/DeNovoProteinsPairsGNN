from pathlib import Path
import json
import random
from typing import List, Optional

import torch
import pytorch_lightning as pl

from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer
from ProteinPairsGenerator.Data import GeneralData
from ProteinPairsGenerator.utils import seq_to_tensor, AMINO_ACIDS_MAP


"""
    Dataset for corrector
"""
class GuessDataset(torch.utils.data.IterableDataset):

    def __init__(self, root : str, batch_size = 0, xToken : str = "guess", yToken : str = "seq", maskToken = "mask") -> None:
        super().__init__()
        self.tokenizer = AdaptedTAPETokenizer()
        self.xToken = xToken
        self.yToken = yToken
        self.maskToken = maskToken
        self.batch_size = batch_size

        self.data = []

        # Iterate over dataset and create dataset with each 
        # batch having roughly batch_size tokens
        x_batch, y_batch, mask_batch = [], [], []
        B, L = 0, 0
        for l in open(root, "r").readlines():

            # Load line
            jsonDict = json.loads(l)

            # Add sequence to batch
            x_batch.append(jsonDict[self.xToken])
            y_batch.append(jsonDict[self.yToken])
            mask_batch.append(jsonDict[self.maskToken])

            # Update size parameters
            B += 1
            L = max(L, len(x_batch[-1]))

            # Test if batch is big enough
            if B * L > batch_size:
                self.data.append(
                    self.combine_batch(B, L, x_batch, y_batch, mask_batch
                ))
                x_batch, y_batch, mask_batch = [], [], []
                B, L = 0, 0

        # Append last batch if not empty
        if B > 0:
            self.data.append(
                self.combine_batch(B, L, x_batch, y_batch, mask_batch
            ))
            

    def combine_batch(self, B, L, x_batch, y_batch, mask_batch):

        x = torch.zeros(B, L, dtype=torch.long)
        y = torch.zeros(B, L, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.bool)

        for i in range(B):
            length = len(x_batch[i])
            x[i, :length] = seq_to_tensor(x_batch[i], AMINO_ACIDS_MAP)
            y[i, :length] = seq_to_tensor(y_batch[i], AMINO_ACIDS_MAP)
            mask[i, :length] = torch.BoolTensor(mask_batch[i])

        return GeneralData(
          x = x,
          y = y,
          mask = mask
        )


    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


"""
    Data loader for corrector
"""
class GuessLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, num_workers=0, shuffle=True):

        if shuffle:
            random.shuffle(dataset.data)

        super().__init__(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])


"""
    Data module for corrector
"""
class GuessDataModule(pl.LightningDataModule):

    def __init__(
        self,
        trainSet: Optional[str] = None,
        valSet: Optional[str] = None,
        testSet: Optional[str] = None,
        batch_size = 0,
        num_workers=0
    ) -> None:
        super().__init__()

        self.trainDataset = GuessDataset(trainSet, batch_size) if trainSet else None
        self.valDataset = GuessDataset(valSet, batch_size) if valSet else None
        self.testDataset = GuessDataset(testSet, batch_size) if testSet else None

        self.num_workers = num_workers

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return GuessLoader(
            dataset=self.trainDataset,
            shuffle=True,
            num_workers=self.num_workers
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
