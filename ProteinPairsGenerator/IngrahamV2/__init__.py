from .model import Struct2Seq as IngrahamV2Model
from .model2 import Struct2Seq as IngrahamV3Model
from ProteinPairsGenerator.DistilationKnowledge import getKDModel
IngrahamV3ModelKD = getKDModel(IngrahamV3Model, -1)
