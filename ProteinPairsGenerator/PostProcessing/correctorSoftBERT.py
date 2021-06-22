# General imports
from tape import ProteinBertModel
from tape.models.modeling_utils  import MLMHead

# Pytorch imports
import torch
from torch import nn
import pytorch_lightning as pl

# Local imports
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer
from ProteinPairsGenerator.utils import AMINO_ACID_NULL, AMINO_ACIDS_MAP

class CorrectorSoftBERT(pl.LightningModule):

    def __init__(
        self, 
        hidden_size : int,
        input_size : int,
        N : int,
        dropout : float
    ) -> None:
        super().__init__()

        self.detector = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=N, dropout=dropout, bidirectional=True)
        self.switch = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Sigmoid()
        )
    
    def corrector(self, x):
        raise NotImplementedError
    
    def masker(self, x):
        raise NotImplementedError

    def forward(self, x):

        print("x dim: ", x.shape)

        h, _ = self.detector(x)

        
        print("h dim: ", h.shape)

        p = self.switch(h)

        print("p dim: ", p.shape)
        print("m dim: ", self.masker(x).shape)

        x_new = p * self.masker(x) + (1 - p) * x
        return self.corrector(x_new)


class CorrectorFullSoftBERT(CorrectorSoftBERT):

    def __init__(
        self, 
        hidden_size : int,
        N : int,
        dropout : float
    ) -> None:
        super().__init__(hidden_size=hidden_size, input_size=768, N=N, dropout=dropout)

        self.tokenizer = AdaptedTAPETokenizer()
        self.bert = ProteinBertModel.from_pretrained('bert-base')
        self.mlm = MLMHead(
            hidden_size=hidden_size, 
            vocab_size=20,
            ignore_index=-1
        )

        """ Make sure we are sharing the input and output embeddings. 
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self.bert._tie_or_clone_weights(self.mlm.decoder, self.bert.embeddings.word_embeddings)

    def corrector(self, x):
        return self.mlm(x)[0]

    def masker(self, x):
        nullSeq = torch.empty_like(x[:, :, 0].squeeze(-1), requires_grad=False)
        nullSeq.fill_(AMINO_ACIDS_MAP[AMINO_ACID_NULL])
        return self.bert(self.tokenizer.AA2BERT(nullSeq))[0]

    def forward(self, x):
        xEmbed = self.bert(self.tokenizer.AA2BERT(x))[0]
        return super().forward(xEmbed)
