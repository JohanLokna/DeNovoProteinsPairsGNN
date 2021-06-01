# General imports
from typing import Union
import warnings

# Pytorch imports
import torch
from torch.utils.data import Subset

# Local imports
from ProteinPairsGenerator.Data import BERTLoader
from ProteinPairsGenerator.BERTModel import maskBERT

class StrokachLoader(BERTLoader):

    def __init__(
        self,
        dataset : Subset,
        teacher : Union[str, None] = None,
        *args,
        **kwargs
    ) -> None:

        self.teacher = teacher
        self.validFields = ["seq", "edge_index", "edge_attr", "maskBERT"]
        if self.teacher:
            self.validFields.append(self.teacher)

        # Ensure correct kwargs
        if not (kwargs.pop("batch_size", None) in [1, None]):
            warnings.warn("Batch size can only be 1. Continuing with batch_size = 1.")
        kwargs["batch_size"] = 1
        kwargs["collate_fn"] = self.updateElement

        super().__init__(
            dataset=dataset,
            *args,
            **kwargs
        )

    def updateElement(self, x):

            # Extracting data from list and only use valid fields
            x = x[0]
            x.__dict__ = {k: v for k, v in x.__dict__.items() if k in self.validFields}

            # Randomly selecet masked sequence
            idx = torch.randint(len(x.maskBERT), (1,)).item()
            maskBERTList = x.__dict__.pop("maskBERT")
            x.maskedSeq, x.mask = maskBERTList[idx]
            if self.teacher and self.teacher in x.__dict__.keys():
                x.__dict__[self.teacher] = x.__dict__[self.teacher][idx]
                x.__dict__["teacherLabels"] = x.__dict__.pop(self.teacher)

            return x
