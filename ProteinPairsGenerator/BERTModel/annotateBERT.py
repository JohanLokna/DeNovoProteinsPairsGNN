from tape import ProteinBertForMaskedLM

from torch import nn

from .utilsTAPE import AdaptedTAPETokenizer

class Annotator(nn.Module):

    def __init__(
        self,
        model : nn.Module,
        tokenizer
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        for param in self.teacher.parameters():
            param.requires_grad = False

    def __call__(self, inTensors):
        return self.BERT2AA(self.model(self.AA2BERT(inTensors)))

class TAPEAnnotator(Annotator):
    
    def __init__(self):
          super().__init__(ProteinBertForMaskedLM.from_pretrained('bert-base'), AdaptedTAPETokenizer())
