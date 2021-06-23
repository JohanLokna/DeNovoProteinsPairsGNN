# General imports
from typing import List

# Pytorch imports
import pytorch_lightning as pl


class BERTModel(pl.LightningModule):
    def __init__(self, extraLogging : List[tuple] = []) -> None:
        super().__init__()
        self.extraLogging = extraLogging
    
    def configure_optimizers(self):
        raise NotImplementedError
    
    def step(self, batch):
        raise NotImplementedError

    def epoch_end(self, phase : str, outputs):
        nCorrect = sum([o["nCorrect"] for o in outputs])
        nTotal = sum([o["nTotal"]  for o in outputs])
        loss = sum([o["loss"] * o["nTotal"] for o in outputs]) / nTotal
        self.log(phase + "Loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(phase + "Acc", nCorrect / nTotal, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for name, f in self.extraLogging:
            self.log(phase + name, f(outputs), on_step=False, on_epoch=True, prog_bar=False, logger=True)

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
