from tape import ProteinBertForMaskedLM

import torch
from torch import nn

from .utilsTAPE import AdaptedTAPETokenizer


class Annotator(nn.Module):

    def __init__(
        self,
        model : nn.Module,
        tokenizer,
        device : str = "cuda:0" if torch.cuda.is_available() else "cpu"
    ) -> None:
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device=self.device)

    def __call__(self, inTensors):
        inTensors = self.tokenizer.AA2BERT(inTensors).to(device=self.device)
        outTensors = self.model(inTensors)[0]
        return self.tokenizer.BERT2AA(outTensors)


class TAPEAnnotator(Annotator):
    
    def __init__(self):
          super().__init__(ProteinBertForMaskedLM.from_pretrained('bert-base'), AdaptedTAPETokenizer())
