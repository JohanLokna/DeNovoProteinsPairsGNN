from tape import ProteinBertForMaskedLM

from torch import nn

from .utilsTAPE import AdaptedTAPETokenizer


class Annotator(nn.Module):

    def __init__(
        self,
        model : nn.Module,
        tokenizer
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, inTensors):
        inTensors = self.tokenizer.AA2BERT(inTensors).type_as(inTensors)
        outTensors = self.model(inTensors)[0]
        return self.tokenizer.BERT2AA(outTensors).type_as(inTensors)


class TAPEAnnotator(Annotator):
    
    def __init__(self, *args, **kwargs):
          super().__init__(ProteinBertForMaskedLM.from_pretrained('bert-base'), AdaptedTAPETokenizer(), *args, **kwargs)
