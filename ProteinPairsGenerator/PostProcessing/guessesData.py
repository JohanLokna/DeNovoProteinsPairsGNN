from pathlib import Path
import json

import torch

from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer
from ProteinPairsGenerator.Data import GeneralData
from ProteinPairsGenerator.utils import seq_to_tensor, AMINO_ACIDS_MAP

class GuessDataset(torch.utils.data.IterableDataset):

    def __init__(self, root : Path) -> None:
        super().__init__()
        self.tokenizer = AdaptedTAPETokenizer()

        self.data = []
        for l in open(root, "r").readlines():

            jsonDict = json.loads(l)

            seq = seq_to_tensor(jsonDict["seq"], AMINO_ACIDS_MAP)
            guess = seq_to_tensor(jsonDict["guess"], AMINO_ACIDS_MAP)

            self.data.append(GeneralData(
              seq = self.tokenizer.AA2BERT(seq)[0],
              guess = self.tokenizer.AA2BERT(guess)[0]
            ))

    def __iter__(self):
        return iter(self.data)
