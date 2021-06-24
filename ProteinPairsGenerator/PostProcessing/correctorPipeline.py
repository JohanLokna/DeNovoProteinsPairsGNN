# General imports
from copy import deepcopy
from typing import Union

# Pytorch imports
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Local imports
from ProteinPairsGenerator.Data import GeneralData


def getCorrectorPipeline(corrector : pl.LightningModule):

    class CorrectorPipeline(corrector):

        def __init__(self, baseModel : nn.Module, classDim : int, checkpoint : Union[None, str], *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.classDim = classDim
            self.baseModel = baseModel

            # Load model if checkpoint is provided
            if checkpoint:
                cpkt = torch.load(checkpoint)
                self.baseModel.load_state_dict(cpkt["state_dict"])

            # Freeze base model
            for param in self.baseModel.parameters():
                param.requires_grad = False

            # Wrap base model to get output
            self.output = None

            def wrapper(*args, **kwargs):
                baseModelOutput = type(self.baseModel).forward(*args, **kwargs)
                self.output = torch.argmax(baseModelOutput.data, self.classDim)
                return baseModelOutput

            baseModel.forward = wrapper.__get__(baseModel, type(baseModel))

        def step(self, x):

            xOld = deepcopy(x)

            d = self.baseModel.step(x)

            print(d)

            print(xOld.seq)
            print(x.seq)
            print(self.output)

            return super().step(GeneralData(x=self.output, y=x.seq))

    return CorrectorPipeline
