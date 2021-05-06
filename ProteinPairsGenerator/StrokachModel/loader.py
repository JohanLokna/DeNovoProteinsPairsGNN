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
            for key in x.__dict__.keys():
                if not key in self.validFields:
                    x.__dict__.pop(key)
            maskedSeq, mask = maskBERT(x.seq, self.subMatirx)
            x.maskedSeq = maskedSeq
            x.mask = mask
            return x

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: updateElement
        )
