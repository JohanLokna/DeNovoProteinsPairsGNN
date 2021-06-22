import numpy as np
from typing import Union, List

import torch

from ProteinPairsGenerator.utils import tensor_to_seq, AMINO_ACID_NULL, AMINO_ACIDS_MAP, AMINO_ACIDS_BASE
from tape.tokenizers import TAPETokenizer, IUPAC_VOCAB

class AdaptedTAPETokenizer(TAPETokenizer):

    AAColsBERT = [IUPAC_VOCAB[aa] for aa in AMINO_ACIDS_BASE]
    findValue = IUPAC_VOCAB[AMINO_ACID_NULL]
    replaceValue = IUPAC_VOCAB["<mask>"]

    def __init__(self):
        super().__init__("iupac")

    def AA2BERT(self, inTensors: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:

        if isinstance(inTensors, torch.Tensor):
            inTensors = [inTensors]

        B, LMax = len(inTensors), max([torch.numel(seq) + 2 for seq in inTensors])
        tokenizedTensos = torch.zeros(B, LMax, dtype=torch.long)

        for i, seq in enumerate(inTensors):
            enc = super().encode(tensor_to_seq(seq, AMINO_ACIDS_MAP))
            tokenizedTensos[i] = torch.from_numpy(np.where(enc == self.findValue, self.replaceValue, enc))

        return tokenizedTensos

    def BERT2AA(self, inTensors: torch.Tensor) -> torch.Tensor:
        return inTensors[:, 1:-1, self.AAColsBERT]
