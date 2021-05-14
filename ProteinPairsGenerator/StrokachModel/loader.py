# General imports
from typing import Union

# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset

class StrokachLoader(DataLoader):

    def __init__(
        self,
        dataset : Dataset,
        teacher : Union[str, None] = None,
        *args,
        **kwargs
    ) -> None:

        self.teacher = teacher
        self.validFields = ["seq", "edge_index", "edge_attr", "maskBERT"]
        if self.teacher:
            self.validFields.append(self.teacher)

        def updateElement(x):

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

        super().__init__(
            dataset=dataset,
            collate_fn=updateElement,
            batch_size = 1,
            *args,
            **kwargs
        )
