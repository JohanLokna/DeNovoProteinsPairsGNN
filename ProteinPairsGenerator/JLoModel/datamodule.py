from ProteinPairsGenerator.Data import BERTDataModule
from .loader import JLoLoader

class JLoDataModule(BERTDataModule):

    def __init__(self, *args, **kwargs):
          kwargs["loaderClass"] = JLoLoader
          super().__init__(*args, **kwargs)
