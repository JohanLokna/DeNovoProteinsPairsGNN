# General imports
from tape import ProteinBertModel
from tape.models.modeling_utils  import MLMHead

# Pytorch imports
import torch
from torch import nn
from torch import optim

# Local imports
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer, BERTModel
from ProteinPairsGenerator.utils import AMINO_ACID_NULL, AMINO_ACIDS_MAP

class CorrectorSoftBERT(BERTModel):

    def __init__(
        self, 
        hidden_size : int,
        input_size : int,
        N : int,
        dropout : float,
        alpha : float
    ) -> None:
        super().__init__()

        self.detector = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=N, dropout=dropout, bidirectional=True)
        self.switch = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Sigmoid()
        )

        # Set up criterion
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def corrector(self, x):
        raise NotImplementedError
    
    def masker(self, x):
        raise NotImplementedError

    def criterion(self, p : torch.Tensor, yHat : torch.Tensor, x : torch.Tensor, y : torch.Tensor):
        g = (x != y).type_as(p)
        return self.alpha * self.bce(p, g) + (1 - self.alpha) * self.ce(yHat, y)

    def forward(self, x):
        h, _ = self.detector(x)
        p = self.switch(h)
        x_new = p * self.masker(x) + (1 - p) * x
        return self.corrector(x_new), p

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'valLoss'
       }
    
    def step(self, x):

        yHat, p = self(x.x)
        loss = self.criterion(p, yHat, x.x, x.y)

        yPred = torch.argmax(yHat.data, 1)
        nCorrect = (yPred == x.y).sum()
        nTotal = torch.numel(yPred)

        return {
            "loss" : loss,
            "nCorrect" : nCorrect,
            "nTotal" : nTotal
        }


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
            hidden_size=self.bert.config.hidden_size, 
            vocab_size=self.bert.config.vocab_size,
            ignore_index=-1
        )

        # Freeze BERT embedding
        for param in self.bert.parameters():
            param.requires_grad = False

        """ Make sure we are sharing the input and output embeddings. 
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self.bert._tie_or_clone_weights(self.mlm.decoder, self.bert.embeddings.word_embeddings)

    def corrector(self, x):
        return self.tokenizer.BERT2AA(self.mlm(x)[0])

    def masker(self, x):
        nullSeq = torch.empty_like(x[:, :, 0].squeeze(-1), requires_grad=False)
        nullSeq.fill_(AMINO_ACIDS_MAP[AMINO_ACID_NULL])
        return self.bert(self.tokenizer.AA2BERT(nullSeq))[0][:, 1:-1]

    def forward(self, x):
        xEmbed = self.bert(self.tokenizer.AA2BERT(x))[0][:, 1:-1]
        return super().forward(xEmbed)
