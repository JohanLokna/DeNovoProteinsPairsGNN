# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset

# Local imports
from ProteinPairsGenerator.utils import maskBERT

class StrokachLoader(DataLoader):

    def __init__(
        self,
        dataset : Dataset,
        batch_size : int
    ) -> None:

        self.nTokens = 20
        self.subMatirx = torch.ones(self.nTokens, self.nTokens)

        def singleElement(x):
            maksedSeq, mask = maskBERT(x.seq, self.subMatirx)
            return tuple((maksedSeq, x.edge_index, x.edge_attr)), x.seq, mask

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: [singleElement(x) for x in batch]
        )
