from .model import Net as JLoModel
from .datamodule import *
from .loader import *
from ProteinPairsGenerator.DistilationKnowledge import getKDModel
JLoModelKD = getKDModel(JLoModel, 1)
