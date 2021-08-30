import pandas as pd
from typing import Optional
from pathlib import Path
import os

import torch

blosumPath = Path(os.path.dirname(os.path.realpath(__file__))).joinpath("BLOSUM/blosum62.csv")

class ScoreBLOSUM(torch.nn.Module):

    def __init__(
        self,
    ) -> None:
        super().__init__()
        a = pd.read_csv(open(blosumPath, "r"))
        print(a)
        B : torch.Tensor = torch.from_numpy(a.values)
        self.B = torch.transpose(B, 0, 1)

    def forward(self, y_true : torch.Tensor, y_pred : torch.Tensor, mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.sum(self.B[y_true.flatten(), :] * y_pred.reshape((-1, y_pred.shape[-1])))
