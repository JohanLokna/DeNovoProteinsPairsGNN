from .model import Net as StrokachModel
from .datamodule import StrokachDataModule
from .loader import StrokachLoader
from ProteinPairsGenerator.DistilationKnowledge import getKDModel
StrokachModelKD = getKDModel(StrokachModel, 1)
