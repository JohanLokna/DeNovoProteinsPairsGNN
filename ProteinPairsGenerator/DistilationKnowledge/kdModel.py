# Pytorch imports
import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
import pytorch_lightning as pl

"""
    Function which creates a wrapper class from the base model which can be used to train with knowledge distillation
"""
def getKDModel(baseModel : pl.LightningModule, classDim : int):

    class KDModel(baseModel):

        def __init__(
            self,
            alpha : float = 0.0,
            *args, 
            **kwargs
        ) -> None:
            super().__init__(*args, **kwargs)
            self.alpha = alpha
            self.output = None
            self.classDim = classDim

        def step(self, x):
            outDict = super().step(x)
            lossTeacher = torch.sum(-log_softmax(self.output, dim=self.classDim) \
                                   * softmax(x.teacherLabels, dim=self.classDim), dim=self.classDim)
            lossTeacher = torch.mean(lossTeacher[x.mask]) if x.mask.dtype in [torch.long, torch.bool] \
                     else torch.sum(lossTeacher * x.mask) / torch.sum(x.mask)
            outDict["loss"] = (1 - self.alpha) * outDict["loss"] + self.alpha * lossTeacher
            return outDict

        def __call__(self, *args, **kwargs):
            self.output = super().__call__(*args, **kwargs)
            return self.output

    return KDModel
