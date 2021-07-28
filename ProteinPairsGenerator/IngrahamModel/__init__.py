from .model2 import Struct2Seq as IngrahamModel
from .loader import IngrahamLoader
from .datamodule import IngrahamDataModule
from ProteinPairsGenerator.DistilationKnowledge import getKDModel
IngrahamModelKD = getKDModel(IngrahamModel, -1)
