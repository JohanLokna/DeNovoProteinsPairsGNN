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
        self.B = B

    def forward(self, y_true : torch.Tensor, y_pred : torch.Tensor, mask : Optional[torch.Tensor] = None) -> torch.Tensor:

        b_scores = self.B[y_true.flatten(), :]
        pred_scores = y_pred.reshape((-1, y_pred.shape[-1]))
        print(b_scores.shape, pred_scores.shape)
        if mask is None:
            return torch.sum(b_scores * pred_scores)
        else:
            return torch.dot(torch.sum(b_scores * pred_scores, -1), mask.flatten())
