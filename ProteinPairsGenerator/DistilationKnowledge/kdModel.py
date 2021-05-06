import torch
import torch.nn as nn
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
            self.sofmax = nn.Softmax(dim=1)
            self.criterion = lambda out, lbl: -torch.sum(self.sofmax(out) * lbl)

        def step(self, batch):
            outDict = super().step(batch)
            outDict["loss"] = self.alpha * outDict["loss"] \
                            + (1 - self.alpha) * self.criterion(self.output, batch.teacherSeq)
            return outDict

        def __call__(self, *args, **kwargs):
            self.output = super().__call__(*args, **kwargs)
            return self.output

    return KDModel
