from pathlib import Path
import json
from random import shuffle

import torch

from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer
from ProteinPairsGenerator.Data import GeneralData
from ProteinPairsGenerator.utils import seq_to_tensor, AMINO_ACIDS_MAP

class GuessDataset(torch.utils.data.IterableDataset):

    def __init__(self, root : Path, xToken : str = "seq", yToken : str = "guess") -> None:
        super().__init__()
        self.tokenizer = AdaptedTAPETokenizer()
        self.xToken = xToken
        self.yToken = yToken

        self.data = []
        for l in open(root, "r").readlines():

            jsonDict = json.loads(l)

            x = seq_to_tensor(jsonDict[self.xToken], AMINO_ACIDS_MAP)
            y = seq_to_tensor(jsonDict[self.yToken], AMINO_ACIDS_MAP)

            self.data.append(GeneralData(
              x = self.tokenizer.AA2BERT(x)[0],
              y = self.tokenizer.AA2BERT(y)[0]
            ))

    def __iter__(self):
        return iter(self.data)

class GuessLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        super().__init__(dataset, batch_size, shuffle, num_workers, collate_fn=lambda x: x)
