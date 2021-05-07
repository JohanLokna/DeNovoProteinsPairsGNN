import numpy as np

import torch

from ProteinPairsGenerator.utils import tensor_to_seq, AMINO_ACID_NULL, AMINO_ACIDS_MAP, AMINO_ACIDS_BASE
from tape.tokenizers import TAPETokenizer, IUPAC_VOCAB

outColumnsFromBERT = [IUPAC_VOCAB[aa] for aa in AMINO_ACIDS_BASE]

class KDTokenizer(TAPETokenizer):

    def __init__(self):
        super().__init__("iupac")
        self.findValue = IUPAC_VOCAB[AMINO_ACID_NULL]
        self.replaceValue = IUPAC_VOCAB["<mask>"]

    def encode(self, inTensor: torch.Tensor) -> torch.Tensor:
      encoded = super().encode(tensor_to_seq(inTensor, AMINO_ACIDS_MAP))
      return torch.from_numpy(np.where(encoded == self.findValue, self.replaceValue, encoded)).unsqueeze(0)

def extractBaseAcids(inTensor: torch.Tensor) -> torch.Tensor:
    print(inTensor.shape)
    return inTensor[0, :, outColumnsFromBERT]
