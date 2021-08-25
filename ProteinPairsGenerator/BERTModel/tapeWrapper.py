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
        seq_bert = self.tokenizer.AA2BERT(seq).type_as(seq)

        # Compute using bert
        out = self.model(seq_bert)[0]

        # Return in AA representation
        return self.tokenizer.BERT2AA(out)

    # Simple step as no training is performed
    def step(self, x):      
        output = self(x.maskedSeq)

        # Only keep masked fraction
        output = output[x.mask.reshape(output.shape[:-1])]
        yTrue = x.seq[x.mask]

        loss = self.criterion(output, yTrue)

        nTotal = torch.numel(yTrue)
        out = {
            "loss" : loss,
            "nTotal" : nTotal
        }
        
        for k in self.kAccuracy:
            yPred_k = torch.topk(output.data, k, -1).indices
            nCorrect_k = (yTrue.unsqueeze(-1) == yPred_k).sum()
            out.update({"nCorrect_{}".format(k): nCorrect_k})

        return out
