# Pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset

# Local imports
from ProteinPairsGenerator.utils import maskBERT
from ProteinPairsGenerator.utils import AMINO_ACIDS_BASE, AMINO_ACID_NULL

class IngrahamLoader(DataLoader):

    def __init__(
        self,
        dataset : Dataset,
        batch_size : int
    ) -> None:

        self.nTokens = len(AMINO_ACIDS_BASE + [AMINO_ACID_NULL])
        self.subMatirx = torch.ones(self.nTokens, self.nTokens)

        def featurize(batch):

            B = len(batch)
            lengths = [torch.numel(b.seq) for b in batch]
            L_max = max(lengths)
            X = torch.zeros(B, L_max, 3, 3)
            S = torch.zeros(B, L_max, dtype=torch.long)
            SBERT = torch.zeros(B, L_max, dtype=torch.long)
            valid = torch.zeros(B, L_max)
            validBERT = torch.zeros(B, L_max)

            # Build the batch
            for i, (b, l) in enumerate(zip(batch, lengths)):
                SBERT[i, :l], validBERT[i, :l] = maskBERT(b.seq, self.subMatirx)
                X[i, :l] = torch.stack([b.coordsN, b.coordsCA, b.coordsC], dim=1)
                S[i, :l] = b.seq
                valid[i, :l] = 1.0

            return tuple((X, SBERT, lengths, valid)), S, validBERT

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=featurize
        )
