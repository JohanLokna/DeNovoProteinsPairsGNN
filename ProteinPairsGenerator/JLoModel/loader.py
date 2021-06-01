# General imports
from typing import Union
import warnings

# Pytorch imports
import torch
from torch.utils.data import Subset

# Local imports
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer
from ProteinPairsGenerator.StrokachModel import StrokachLoader

class JLoLoader(StrokachLoader):

    def __init__(self, dataset: Subset, teacher: Union[str, None], *args, **kwargs) -> None:
        super().__init__(dataset, teacher=teacher, *args, **kwargs)
        self.tokenizer = AdaptedTAPETokenizer()

    def updateElement(self, x):
        x = super().updateElement(x)
        x.maskedSeq = self.tokenizer.AA2BERT(x.maskedSeq)[0]
        return x
