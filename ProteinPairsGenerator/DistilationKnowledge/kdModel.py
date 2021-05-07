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
            self.softmax = nn.Softmax(dim=1)

        def step(self, x):
            outDict = super().step(x)

            l = torch.mean(torch.sum(-torch.log_softmax(self.output, dim=1) * x.teacherLabels, dim=1)[self.mask])

            print(l, outDict["loss"])

            outDict["loss"] = self.alpha * outDict["loss"] \
            + (1 - self.alpha) * l

            return outDict

        def __call__(self, *args, **kwargs):
            self.output = super().__call__(*args, **kwargs)
            return self.output

    return KDModel
