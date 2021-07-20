from .struct2seq import Struct2Seq as IngrahamModel
from .loader import IngrahamLoader
from .datamodule import IngrahamDataModule
from ProteinPairsGenerator.DistilationKnowledge import getKDModel
IngrahamModelKD = getKDModel(IngrahamModel, -1)
