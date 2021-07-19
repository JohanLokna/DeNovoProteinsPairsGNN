# General imports
from typing import Union

# Pytorch imports
import torch
from torch.utils.data import Subset

# Local imports
from ProteinPairsGenerator.Data import GeneralData
from ProteinPairsGenerator.utils import AMINO_ACIDS_BASE
from ProteinPairsGenerator.Data import BERTLoader


class IngrahamLoader(BERTLoader):

    def __init__(
        self,
        dataset : Subset,
        teacher : Union[str, None] = None,
        *args,
        **kwargs
    ) -> None:

        self.nTokens = len(AMINO_ACIDS_BASE)
        self.teacher = teacher

        def featurize(batch):

            B = len(batch)
            lengths = [torch.numel(b.seq) for b in batch]
            L_max = max(lengths)
            coords = torch.zeros(B, L_max, 3, 3)
            seq = torch.zeros(B, L_max, dtype=torch.long)
            maskedSeq = torch.zeros(B, L_max, dtype=torch.long)
            mask = torch.zeros(B, L_max)
            valid = torch.zeros(B, L_max)

            if self.teacher:
                teacherLabels = torch.zeros(B, L_max, self.nTokens)

            # Build the batch
            for i, (b, l) in enumerate(zip(batch, lengths)):

                # Standard features
                coords[i, :l] = b.coords / 100 # Fix this
                seq[i, :l] = b.seq
                valid[i, :l] = 1.0

                # Randomly selecet masked sequence
                idx = torch.randint(len(b.maskBERT), (1,)).item()
                maskBERTList =b["maskBERT"]
                maskedSeq[i, :l], mask[i, :l] = maskBERTList[idx]
                if self.teacher:
                    teacherLabels[i, :l] = b.__dict__[self.teacher][idx]

            return GeneralData(
                coords=coords,
                seq=seq,
                valid=valid,
                lengths=lengths,
                maskedSeq=maskedSeq,
                mask=mask,
                **({"teacherLabels": teacherLabels} if self.teacher else {})
            )

        # Ensure correct kwargs
        kwargs["collate_fn"] = featurize

        super().__init__(
            dataset=dataset,
            *args,
            **kwargs
        )
