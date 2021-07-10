# General imports
from pathlib import Path
from typing import Union, List

# Local imports
from ProteinPairsGenerator.Data import BERTDataModule
from .loader import JLoLoader

class JLoDataModule(BERTDataModule):

    def __init__(
        self,
        root : Union[Path, str],
        trainSet: Union[Path, List[Path], None] = None,
        valSet: Union[List[Path], None] = None,
        testSet: Union[List[Path], None] = None,
        teacher: Union[str, None] = None,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        batch_size: int = 1
    ):
          super().__init__(
              root,
              JLoLoader,
              trainSet,
              valSet,
              testSet,
              teacher,
              num_workers,
              prefetch_factor,
              batch_size
          )
