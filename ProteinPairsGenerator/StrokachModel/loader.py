# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset

# Local imports
from ProteinPairsGenerator.utils import maskBERT

class StrokachLoader(DataLoader):

    def __init__(
        self,
        dataset : Dataset,
        tokenizer = None
    ) -> None:

        self.nTokens = 20
        self.subMatirx = torch.ones(self.nTokens, self.nTokens)
        self.validFields = ["seq", "edge_index", "edge_attr"]
        self.tokenizer = tokenizer

        def updateElement(x):
            x = x[0]
            x.__dict__ = {k: v for k, v in x.__dict__.items() if k in self.validFields}
            maskedSeq, mask = maskBERT(x.seq, self.subMatirx)
            x.maskedSeq = maskedSeq
            x.mask = mask
            if not self.tokenizer is None:
                x.tokenizedSeq = self.tokenizer.encode(x.seq)
            return x

        super().__init__(
            dataset=dataset,
            collate_fn=updateElement
        )
