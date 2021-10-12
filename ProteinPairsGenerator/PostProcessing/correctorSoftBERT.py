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

"""
    Base model class for soft masked bert corrector
"""
class CorrectorSoftBERT(BERTModel):

    def __init__(
        self, 
        hidden_size : int,
        input_size : int,
        N : int,
        dropout : float,
        alpha : float,
        lr : float
    ) -> None:

        extraLogging = [
          ("R2R", lambda outputs: sum([o["R2R"]  for o in outputs]) / sum([o["nTotal"]  for o in outputs])),
          ("R2W", lambda outputs: sum([o["R2W"]  for o in outputs]) / sum([o["nTotal"]  for o in outputs])),
          ("W2R", lambda outputs: sum([o["W2R"]  for o in outputs]) / sum([o["nTotal"]  for o in outputs])),
          ("W2W", lambda outputs: sum([o["W2W"]  for o in outputs]) / sum([o["nTotal"]  for o in outputs]))
        ]

        super().__init__(extraLogging=extraLogging, kAccuracy=[])

        self.lrInitial = lr

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
        return self.alpha * self.bce(p.flatten(), g.flatten()) \
             + (1 - self.alpha) * self.ce(yHat.view((-1, yHat.shape[-1])), y.flatten())

    def forward(self, x):
        h, _ = self.detector(x)
        p = self.switch(h)
        x_new = p * self.masker(x) + (1 - p) * x
        return self.corrector(x_new), p.squeeze(-1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lrInitial)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'valLoss'
       }
    
    def step(self, input, extra_out={}):

        # Get results and use mask
        yHat, p = self(input.x)
        adaptedMask = input.mask if len(input.mask.shape) == 2 else input.mask.unsqueeze(0)
        p, yHat = p[adaptedMask], yHat[adaptedMask], 
        x, y = input.x[input.mask], input.y[input.mask]

        # Compute loss
        loss = self.criterion(p, yHat, x, y)

        # Compute standard metrics
        yPred = torch.argmax(yHat.data, -1)
        nCorrect = (yPred == y).sum()
        nTotal = torch.numel(yPred)

        # Compute extra metrics
        r2r = ((x == y) & (yPred == y)).sum()
        r2w = ((x == y) & (yPred != y)).sum()
        w2r = ((x != y) & (yPred == y)).sum()
        w2w = ((x != y) & (yPred != y)).sum()

        out = {
            "loss" : loss,
            "nCorrect" : nCorrect,
            "nTotal" : nTotal,
            "R2R" : r2r,
            "R2W" : r2w,
            "W2R" : w2r,
            "W2W" : w2w
        }

        for key, f in extra_out.items():
            out[key] = f(yTrue=y, yHat=yHat, yPred=yPred)

        return out


"""
    Specialization of corrector as used in thesis. Uses TAPE embedding and MLM head
"""
class CorrectorFullSoftBERT(CorrectorSoftBERT):

    def __init__(
        self, 
        hidden_size : int,
        N : int,
        dropout : float,
        alpha : float,
        lr : float
    ) -> None:
        super().__init__(hidden_size=hidden_size, input_size=768, N=N, dropout=dropout, alpha=alpha, lr=lr)

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
        return self.tokenizer.BERT2AA(self.mlm(x)[0]).to(self.device)

    def masker(self, x):
        nullSeq = torch.empty_like(x[:, 1:-1, 0].squeeze(-1), requires_grad=False)
        nullSeq.fill_(AMINO_ACIDS_MAP[AMINO_ACID_NULL])
        return self.bert(self.tokenizer.AA2BERT(nullSeq).to(self.device))[0]

    def forward(self, x):
        xEmbed = self.bert(self.tokenizer.AA2BERT(x).to(self.device))[0]
        output, p = super().forward(xEmbed)
        return output, p[:, 1:-1]
