# General imports
from pathlib import Path
from typing import Union, Optional, List

# Torch imports
import torch
from torch.utils.data import Subset

# Local imports
from ProteinPairsGenerator.Data import BERTDataModule
from .loader import IngrahamLoader

def batchAccordingToSize(subset : Subset, cutoffSize : int = 5000):
    newIndices = []

    currCount = 0
    currBatch = []

    for idx in subset.indices:
        print(idx)
        print(subset.dataset.get(idx))
        currCount += torch.numel(subset.dataset.get(idx).seq)
        currBatch.append(idx)

        if cutoffSize >= cutoffSize:
            newIndices.append(currBatch)
            currCount, currBatch = 0, []

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
        prefetch_factor: int = 2
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
              batch_builder = batchAccordingToSize
          )
