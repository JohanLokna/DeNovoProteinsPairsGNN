import pandas as pd
from typing import Optional

import torch

class ScoreBLOSUM(torch.nn.Module):

    def __init__(
        self, 
        B : torch.Tensor = torch.from_numpy(pd.read_csv(open(".BLOSUM/blosum62.csv", "r")).data)
    ) -> None:
        super().__init__()
        self.B = B

    def forward(self, y_true : torch.Tensor, y_pred : torch.Tensor, mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        
        y_true = y_true.flatten()
        y_pred = y_pred
