import torch
import torch.nn as nn
import pytorch_lightning as pl

from tape import ProteinBertForMaskedLM, TAPETokenizer
tokenizer = TAPETokenizer(vocab='iupac')

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
            self.teacher = ProteinBertForMaskedLM.from_pretrained('bert-base')

        def step(self, x):
            outDict = super().step(x)
            teacherLabels = torch.nn.functional.one_hot(x.seq, num_classes=20).float()
            self.teacher(x.tokenizedSeq)
            lossTeacher = torch.mean(torch.sum(-torch.log_softmax(self.output, dim=1) * teacherLabels, dim=1)[x.mask])
            outDict["loss"] = self.alpha * outDict["loss"] + (1 - self.alpha) * lossTeacher
            return outDict

        def __call__(self, *args, **kwargs):
            self.output = super().__call__(*args, **kwargs)
            return self.output

    return KDModel
