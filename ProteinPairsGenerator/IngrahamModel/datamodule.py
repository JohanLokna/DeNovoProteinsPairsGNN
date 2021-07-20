# General imports
from pathlib import Path
from typing import Union, Optional, List

# Torch imports
import torch

# Local imports
from ProteinPairsGenerator.Data import BERTDataModule
from .loader import IngrahamLoader

def batchAccordingToSize(dataset, indices, cutoffSize = 5000):
    newIndices = []

    currCount = 0
    currBatch = []

    for idx in indices:
        print(indices)
        print(idx)
        print(dataset.get(idx))
        currCount += torch.numel(dataset.get(idx))
        currBatch.append(idx)

        if cutoffSize >= cutoffSize:
            newIndices.append(currBatch)
            currCount, currBatch = 0, []

    return newIndices


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
