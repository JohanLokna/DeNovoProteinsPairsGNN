# General imports
from typing import Union

# Pytorch imports
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# Local imports
from ProteinPairsGenerator.Data import GeneralData
from ProteinPairsGenerator.utils import AMINO_ACIDS_BASE

class IngrahamLoader(DataLoader):

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
            mask = torch.zeros(B, L_max, dtype=torch.long)
            valid = torch.zeros(B, L_max)

            if self.teacher:
                teacherLabels = torch.zeros(B, L_max, self.nTokens)

            # Build the batch
            for i, (b, l) in enumerate(zip(batch, lengths)):

                # Standard features
                coords[i, :l] = b.coords
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
                maskedSeq=maskedSeq,
                mask=mask,
                **({self.teacher: teacherLabels} if self.teacher else {})
            )

        # Ensure correct kwargs
        kwargs["collate_fn"] = featurize

        if ("sampler" in kwargs) and isinstance(kwargs["sampler"], DistributedSampler):
            rank = kwargs["sampler"].rank
            size = kwargs["sampler"].num_replicas
            newIndecies = [x for x in dataset.indices if x[0] % size == rank]
            dataset = Subset(dataset=dataset.dataset, indices=newIndecies)
            kwargs["sampler"] = None
        kwargs["shuffle"] = False

        super().__init__(
            dataset=dataset,
            *args,
            **kwargs
        )
