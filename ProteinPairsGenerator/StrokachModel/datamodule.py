from ProteinPairsGenerator.Data import BERTDataModule
from .loader import StrokachLoader

class StrokachDataModule(BERTDataModule):

    def __init__(self, *args, **kwargs):
          kwargs["loaderClass"] = StrokachLoader
          super().__init__(*args, **kwargs)
