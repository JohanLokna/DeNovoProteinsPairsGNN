# General imports
from tape import ProteinBertModel

# Torch imports
from torch import nn

# Local imports
from ProteinPairsGenerator.StrokachModel import StrokachModel
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer


class BERTHelper(nn.Module):

    def __init__(self, hidden_size : int, use_tokenizer = False):
        super().__init__()
        self.model = ProteinBertModel.from_pretrained("bert-base", cache_dir="/mnt/ds3lab-scratch/jlokna/bertBase")
        for param in self.model.parameters():
            param.requires_grad = False
        self.encode = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        if use_tokenizer:
            self.tokenizer = AdaptedTAPETokenizer()

    def forward(self, x):

        if hasattr(self, "tokenizer"):
            x = self.tokenizer.AA2BERT(x).type_as(x)[0]
        
        return self.encode(self.model(x.unsqueeze(0))[0][0, 1:-1])


class Net(StrokachModel):
    def __init__(
        self,
        use_tokenizer = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_x = BERTHelper(kwargs["hidden_size"], use_tokenizer)

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     state_dict["embed_x.encode.2.weight"] = state_dict.pop("embed_x.encode.1.weight")
    #     state_dict["embed_x.encode.2.bias"] = state_dict.pop("embed_x.encode.1.bias")
    #     super().load_state_dict(state_dict, *args, **kwargs)
