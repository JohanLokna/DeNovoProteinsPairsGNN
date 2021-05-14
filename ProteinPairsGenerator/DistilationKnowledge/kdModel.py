# Pytorch imports
import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
import pytorch_lightning as pl


def getKDModel(baseModel : pl.LightningModule, alpha : float):

    class KDModel(baseModel):

        def __init__(
            self,
            *args, 
            **kwargs
        ) -> None:
            super().__init__(*args, **kwargs)
            self.alpha = alpha
            self.output = None

        def step(self, x):
            print(x)
            outDict = super().step(x)
            lossTeacher = torch.mean(torch.sum(-log_softmax(self.output, dim=1) * softmax(x.teacherLabels, dim=1), dim=1)[x.mask])
            outDict["loss"] = (1 - self.alpha) * outDict["loss"] + self.alpha * lossTeacher
            return outDict

        def __call__(self, *args, **kwargs):
            self.output = super().__call__(*args, **kwargs)
            return self.output

    return KDModel
