from ProteinPairsGenerator.Data import BERTDataModule
from .loader import StrokachLoader

class IngrahamDataModule(BERTDataModule):

    def __init__(self, *args, **kwargs):
          kwargs["loaderClass"] = StrokachLoader
          super().__init__(*args, **kwargs)
