# General imports
from typing import List

# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset

class StrokachLoader(DataLoader):

    def __init__(
        self,
        dataset : Dataset,
        extraFields : List[str] = []
    ) -> None:

        self.validFields = ["seq", "edge_index", "edge_attr", "maskBERT"] + extraFields

        def updateElement(x):

            # Extracting data from list and only use valid fields
            x = x[0]
            x.__dict__ = {k: v for k, v in x.__dict__.items() if k in self.validFields}

            # Randomly selecet masked sequence
            idx = torch.randint(len(x.maskBERT), (1,)).item()
            maskBERTList = x.__dict__.pop("maskBERT")
            x.maskedSeq, x.mask = maskBERTList[idx]
            for key in [k for k in x.__dict__.keys() if k in ["TAPE"]]:
                x.__dict__[key] = x.__dict__[key][idx]

            return x

        super().__init__(
            dataset=dataset,
            collate_fn=updateElement,
            batch_size=1
        )
