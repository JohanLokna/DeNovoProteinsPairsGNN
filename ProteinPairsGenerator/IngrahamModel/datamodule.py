# General imports
from pathlib import Path
from typing import Union, Optional, List

# Torch imports
import torch
from torch.utils.data import Subset

# Local imports
from ProteinPairsGenerator.Data import BERTDataModule
from .loader import IngrahamLoader

def batchAccordingToSize(subset : Subset, cutoffSize : int = 1500):
    newIndices = []

    currCount = 0
    currBatch = []

    for idx in subset.indices:
        currSize = torch.numel(subset.dataset[idx].seq)

        if currCount + currSize >= cutoffSize:
            newIndices.append(currBatch)
            currCount, currBatch = 0, []

        currBatch.append(idx)
        currCount += currSize

    if currCount > 0:
        newIndices.append(currBatch)

    return Subset(subset.dataset, newIndices)


class IngrahamDataModule(BERTDataModule):

    def __init__(
        self,
        root : Union[Path, str],
        trainSet: Optional[Union[Path, List[Path]]] = None,
        valSet: Optional[List[Path]] = None,
        testSet: Optional[List[Path]] = None,
        teacher: Optional[str] = None,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        cutoffSize: int = 1500
    ):
          super().__init__(
              root,
              IngrahamLoader,
              trainSet,
              valSet,
              testSet,
              teacher,
              num_workers,
              prefetch_factor,
              batch_size = 1,
              batch_builder = lambda x: batchAccordingToSize(x, cutoffSize)
          )
