import pandas as pd
from typing import Optional

import torch

class ScoreBLOSUM(torch.nn.Module):

    def __init__(
        self, 
        B : torch.Tensor = torch.from_numpy(pd.read_csv(open("./BLOSUM/blosum62.csv", "r")).data)
    ) -> None:
        super().__init__()
        self.B = torch.transpose(B, 0, 1)

    def forward(self, y_true : torch.Tensor, y_pred : torch.Tensor, mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.sum(self.B[y_true.flatten(), :] * y_pred.reshape((-1, y_pred.shape[-1])))
