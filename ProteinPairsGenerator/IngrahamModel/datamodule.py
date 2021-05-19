from ProteinPairsGenerator.Data import BERTDataModule
from .loader import IngrahamLoader

class IngrahamDataModule(BERTDataModule):

    def __init__(*args, **kwargs):
          super().__init__(loaderClass=IngrahamLoader, *args, **kwargs)
