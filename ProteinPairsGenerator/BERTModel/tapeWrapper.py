from tape import ProteinBertForMaskedLM

import torch

from .modelBERT import BERTModel
from .utilsTAPE import AdaptedTAPETokenizer

class TAPEWrapper(BERTModel):

    def __init__(self) -> None:
        super().__init__()

        self.model = ProteinBertForMaskedLM.from_pretrained('bert-base')
        self.tokenizer = AdaptedTAPETokenizer()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, seq):

        # Change to BERT representaion
        seq = self.tokenizer.AA2BERT(seq)

        print(seq.device)
        print(self.model.device)

        # Compute using bert
        out = self.model(seq)[0]

        # Return in AA representation
        return self.tokenizer.BERT2AA(out)

    # Simple step as no training is performed
    def step(self, x):
        output = self(x.seq)[x.mask]
        yTrue = x.seq[x.mask]

        loss = self.criterion(output, yTrue)

        yPred = torch.argmax(output.data, 1)
        nCorrect = (yPred == yTrue).sum()
        nTotal = torch.numel(yPred)

        return {
            "loss" : loss,
            "nCorrect" : nCorrect,
            "nTotal" : nTotal
        }

    def to(self, device):
        self.tokenizer.device = device
        self.model.to(device)
