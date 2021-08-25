# General imports
from typing import List

# Pytorch imports
import pytorch_lightning as pl


class BERTModel(pl.LightningModule):
    def __init__(self, extraLogging : List[tuple] = [], kAccuracy : List[int] = [1, 3, 5]) -> None:
        super().__init__()
        self.extraLogging = extraLogging
        self.kAccuracy = kAccuracy
    
    def configure_optimizers(self):
        raise NotImplementedError
    
    def step(self, batch):
        raise NotImplementedError

    def epoch_end(self, phase : str, outputs):
        nTotal = sum([o["nTotal"]  for o in outputs])
        loss = sum([o["loss"] * o["nTotal"] for o in outputs]) / nTotal
        self.log(phase + "Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for k in self.kAccuracy:
            nCorrect = sum([o["nCorrect_{}".format(k)] for o in outputs])
            self.log(phase + "Acc_{}".format(k), nCorrect / nTotal, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for name, f in self.extraLogging:
            self.log(phase + name, f(outputs), 
                     on_step=False, on_epoch=True, 
                     prog_bar=False if phase == "train" else True, logger=True)

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
