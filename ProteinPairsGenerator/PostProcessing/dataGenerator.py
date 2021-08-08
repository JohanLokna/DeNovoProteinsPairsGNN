from tqdm import tqdm
import json

import torch

# Local imports
from ProteinPairsGenerator.BERTModel import maskBERT
from ProteinPairsGenerator.utils import AMINO_ACIDS_BASE
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer

class SampleGenerator:

    def __init__(
        self,
        outPath,
        levels,
        repeats,
        classDim,
        device = "cpu"
    ) -> None:
        self.levels = levels
        self.repeats = repeats
        self.device = device
        self.outFile = open(outPath, "a")
        self.dim = classDim

    def run(self, model, dm, verbose = False) -> None:

        # Set up model and way to get output

        model.to(device=self.device)

        def wrapper(*args, **kwargs):
            baseModelOutput = type(model).forward(*args, **kwargs)
            self.output = torch.argmax(baseModelOutput.data, self.classDim)
            return baseModelOutput

        model.forward = wrapper.__get__(model, type(model))

        # Loop over levels
        for level in self.levels:
            for _ in range(self.repeats):
                for x in tqdm(dm.test_dataloader()) if verbose else dm.test_dataloader():

                    self.remask(x, **level)
                    x = dm.transfer_batch_to_device(x, self.device)
                    model.step(x)

                    json.dump({"seq": x.seq, "guess": self.output}, self.outFile)

    def remask(self, x, **kwargs) -> None:
        raise NotImplementedError


class SampleGeneratorStrokach(SampleGenerator):

    def __init__(self, outPath, levels, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(outPath, levels, repeats, 1, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)


class SampleGeneratorIngrham(SampleGenerator):

    def __init__(self, outPath, levels, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(outPath, levels, repeats, -1, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        for i in range(x.seq.shape[0]):
            l = x.lengths[i]
            x.maskedSeq[i, :l], x.mask[i, :l] = maskBERT(x.seq[i, :l], **kwargs)


class SampleGeneratorJLo(SampleGenerator):

    def __init__(self, outPath, levels, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(outPath, levels, repeats, 1, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)
        x.maskedSeq = self.tokenizer.AA2BERT(x.maskedSeq)[0]
