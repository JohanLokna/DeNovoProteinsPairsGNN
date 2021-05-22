# General imports
from tape import ProteinBertModel

# Torch imports
from torch import nn
from torch.nn.modules.container import ModuleList

# Local imports
from ProteinPairsGenerator.StrokachModel import StrokachModel
from ProteinPairsGenerator.BERTModel import Annotator, AdaptedTAPETokenizer


class UnSqueeze(nn.Module):
    def forward(self, x):
        return x.squeeze(0)


class Net(StrokachModel):
    def __init__(
        self, 
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_x = ModuleList([ProteinBertModel.from_pretrained('bert-base'), UnSqueeze()])
