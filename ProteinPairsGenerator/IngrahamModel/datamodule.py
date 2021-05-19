from ProteinPairsGenerator.Data import BERTDataModule
from .loader import IngrahamLoader

class IngrahamDataModule(BERTDataModule):

    def __init__(self, *args, **kwargs):
          kwargs["loaderClass"] = IngrahamLoader
          super().__init__(*args, **kwargs)
