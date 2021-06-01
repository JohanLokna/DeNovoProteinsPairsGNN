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

    def updateElement(self, x):
        x = super().updateElement(x)
        x.maskedSeq = self.tokenizer.AA2BERT(x.maskedSeq)[0]
        return x
