import numpy as np

import torch

from ProteinPairsGenerator.utils import tensor_to_seq, AMINO_ACID_NULL, AMINO_ACIDS_MAP, AMINO_ACIDS_BASE
from tape.tokenizers import TAPETokenizer, IUPAC_VOCAB

class AdaptedTAPETokenizer(TAPETokenizer):

    AAColsBERT = [IUPAC_VOCAB[aa] for aa in AMINO_ACIDS_BASE]
    findValue = IUPAC_VOCAB[AMINO_ACID_NULL]
    replaceValue = IUPAC_VOCAB["<mask>"]

    def __init__(self):
        super().__init__("iupac")

    def AA2BERT(self, inTensors: torch.Tensor) -> torch.Tensor:
        enc = super().encode(tensor_to_seq(inTensors, AMINO_ACIDS_MAP))
        return torch.from_numpy(np.where(enc == self.findValue, self.replaceValue, enc)).unsqueeze(0)

    def BERT2AA(self, inTensors: torch.Tensor) -> torch.Tensor:
        return inTensors[:, 1:-1, self.AAColsBERT]
