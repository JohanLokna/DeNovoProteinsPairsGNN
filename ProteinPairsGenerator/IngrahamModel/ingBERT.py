# General imports
from tape import ProteinBertModel

# Torch imports
from torch import nn

# Local imports
from ProteinPairsGenerator.IngrahamModel import IngrahamModel
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer

"""
    Wrapper module for the TAPE embedding which is used in order to integrate the TAPE embedding as part of the input
"""
class BERTHelperIngraham(nn.Module):

    def __init__(self, hidden_size : int):
        super().__init__()
        self.model = ProteinBertModel.from_pretrained("bert-base", cache_dir="/mnt/ds3lab-scratch/jlokna/bertBase")
        for param in self.model.parameters():
            param.requires_grad = False
        self.encode = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.tokenizer = AdaptedTAPETokenizer()

    def forward(self, x):
        x = self.tokenizer.AA2BERT(x).type_as(x)
        out = self.encode(self.model(x)[0][:, 1:-1]) # Ingraham ist batched, which Strokach is not
        return out

"""
    Specialization of Ingraham model (Struct2Seq) which includes the TAPE-embedding as part of the input
"""
class Net(IngrahamModel):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.W_s = BERTHelperIngraham(self.hidden_dim)
