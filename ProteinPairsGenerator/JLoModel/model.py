# General imports
from tape import ProteinBertModel

# Torch imports
from torch import nn
from torch.nn.modules.container import ModuleList

# Local imports
from ProteinPairsGenerator.StrokachModel import StrokachModel


class BERTHelper(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = ProteinBertModel.from_pretrained('bert-base')

    def forward(self, x):
        return self.model(x.unsqueeze(0))[0, 1:-1].squeeze(0)


class Net(StrokachModel):
    def __init__(
        self, 
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_x = BERTHelper()