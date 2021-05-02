# Pytorch imports
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl

class BERTModel(pl.LightningModule):
    def __init__(
        self,
        criterion = nn.CrossEntropyLoss()
    ) -> None:
        super().__init__()

        self.criterion = criterion
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'valLoss'
       }
    
    def step(self, batch):
        
        loss = 0
        nCorrect = 0
        nTotal = 0
        for x, y, mask in batch:

            output = self(*x)[mask]
            yTrue = y[mask]

            loss += self.criterion(output, yTrue)

            yPred = torch.argmax(output.data, 1)
            nCorrect += (yPred == yTrue).sum()
            nTotal += torch.numel(yPred)

        return {
            "loss" : loss,
            "nCorrect" : nCorrect,
            "nTotal" : nTotal
        }

    def epoch_end(self, phase : str, outputs):
        nCorrect = sum([o["nCorrect"] for o in outputs])
        nTotal = sum([o["nTotal"]  for o in outputs])
        loss = sum([o["loss"] * o["nTotal"] for o in outputs]) / nTotal
        self.log(phase + "Loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(phase + "Acc", nCorrect / nTotal, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, idx):
        return self.step(batch)
    
    def validation_step(self, batch, idx):
        return self.step(batch)
    
    def test_step(self, batch, idx):
        return self.step(batch)

    def training_epoch_end(self, outputs):
        self.epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        self.epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        self.epoch_end("test", outputs)
