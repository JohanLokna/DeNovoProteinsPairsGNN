# General imports
from tape import ProteinBertModel

# Torch imports
from torch import nn

# Local imports
from ProteinPairsGenerator.StrokachModel import StrokachModel


class BERTHelper(nn.Module):

    def __init__(self, hidden_size : int):
        super().__init__()
        self.model = ProteinBertModel.from_pretrained('bert-base')
        for param in self.model.parameters():
            param.requires_grad = False
        self.encode = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, x):
        return self.encode(self.model(x.unsqueeze(0))[0][0, 1:-1])


class Net(StrokachModel):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_x = BERTHelper(kwargs["hidden_size"])