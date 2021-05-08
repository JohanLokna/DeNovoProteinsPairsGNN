# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset

class StrokachLoader(DataLoader):

    def __init__(
        self,
        dataset : Dataset
    ) -> None:

        self.validFields = ["seq", "edge_index", "edge_attr"]

        def updateElement(x):
            x = x[0]
            x.__dict__ = {k: v for k, v in x.__dict__.items() if k in self.validFields}
            return x

        super().__init__(
            dataset=dataset,
            collate_fn=updateElement,
            batch_size=1
        )
