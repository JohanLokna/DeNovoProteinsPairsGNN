# General imports
from tape import ProteinBertModel

# Local imports
from ProteinPairsGenerator.StrokachModel import StrokachModel
from ProteinPairsGenerator.BERTModel import Annotator, AdaptedTAPETokenizer


class Net(StrokachModel):
    def __init__(
        self, 
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_x = Annotator(ProteinBertModel.from_pretrained('bert-base'), AdaptedTAPETokenizer())

