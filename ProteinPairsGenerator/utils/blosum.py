import pandas as pd
from typing import Optional
from pathlib import Path
import os

import torch

blosumPath = Path(os.path.dirname(os.path.realpath(__file__))).joinpath("BLOSUM/blosum62.csv")

class ScoreBLOSUM(torch.nn.Module):

    def __init__(
        self,
        B : torch.Tensor = torch.from_numpy(pd.read_csv(open(blosumPath, "r"), index_col=0).values)
    ) -> None:
        super().__init__()        
        self.B = torch.transpose(B, 0, 1)

    def forward(self, y_true : torch.Tensor, y_pred : torch.Tensor, mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return torch.sum(self.B[y_true.flatten(), :] * y_pred.reshape((-1, y_pred.shape[-1])))
        else:
            return torch.dot(torch.sum(self.B[y_true.flatten(), :] * y_pred.reshape((-1, y_pred.shape[-1])), -1), mask.flatten())
