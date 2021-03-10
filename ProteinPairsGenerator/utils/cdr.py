from ProteinPairsGenerator.utils import tensor_to_seq

def getLightCDR(seq : str):
    yield slice(20, 108)

def getHeavyCDR(seq : str):
    yield slice(20, 130)
