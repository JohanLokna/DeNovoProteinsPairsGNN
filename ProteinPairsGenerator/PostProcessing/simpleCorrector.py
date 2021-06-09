# General import
from tape import ProteinBertForMaskedLM

import torch
from torch import nn

from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer

# Pytorch imports
import torch
import torch.nn as nn
import pytorch_lightning as pl


def getCorrectorModel(baseModel : pl.LightningModule):

    class CorrectorModel(baseModel):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenizer = AdaptedTAPETokenizer()
            self.corrector = ProteinBertForMaskedLM.from_pretrained('bert-base')

        def forward(self, x):
            preds = torch.argmax(super().__call__(x).data, 1)
            corrPreds = self.corrector(self.tokenizer.AA2BERT(preds))
            return self.tokenizer.BERT2AA(corrPreds)

    return CorrectorModel
