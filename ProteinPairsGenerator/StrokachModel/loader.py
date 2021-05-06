# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset

# Local imports
from ProteinPairsGenerator.utils import maskBERT

class StrokachLoader(DataLoader):

    def __init__(
        self,
        dataset : Dataset
    ) -> None:

        self.nTokens = 20
        self.subMatirx = torch.ones(self.nTokens, self.nTokens)
        self.validFields = ["seq", "edge_index", "edge_attr"]

        def updateElement(x):
            x = x[0]
            for key in x.__dict__.keys():
                if not key in self.validFields:
                    x.__dict__.pop(key)
            maskedSeq, mask = maskBERT(x.seq, self.subMatirx)
            x.maskedSeq = maskedSeq
            x.mask = mask
            return xList

        super().__init__(
            dataset=dataset,
            collate_fn=updateElement
        )
