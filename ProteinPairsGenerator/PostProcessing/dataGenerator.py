from tqdm import tqdm
import json

import torch

# Local imports
from ProteinPairsGenerator.BERTModel import maskBERT
from ProteinPairsGenerator.utils import AMINO_ACIDS_BASE, AMINO_ACIDS_MAP, tensor_to_seq
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer

class SampleGenerator:

    def __init__(
        self,
        outPath,
        repeats,
        classDim,
        device = "cpu"
    ) -> None:
        self.repeats = repeats
        self.device = device
        self.outFile = open(outPath, "a+")
        self.classDim = classDim

    def run(self, model, loader, transfer = lambda x: x, verbose = False, forcedMaskKwargs : dict = {}) -> None:

        # Set up model and way to get output

        model.to(device=self.device)

        def wrapper(*args, **kwargs):
            baseModelOutput = type(model).forward(*args, **kwargs)
            self.output = torch.argmax(baseModelOutput.data, self.classDim)
            return baseModelOutput

        model.forward = wrapper.__get__(model, type(model))

        # Loop over levels
        for _ in range(self.repeats):
            for x in tqdm(loader) if verbose else loader:

                level = {"maskFrac": torch.rand(1).item()}

                # Add provided args
                level.update(forcedMaskKwargs)

                self.remask(x, **level)
                x = transfer(x, self.device)
                model.step(x)

                for s, g, m in self.pairs(x, self.output):
                    d = {"seq": tensor_to_seq(s, AMINO_ACIDS_MAP), "guess": tensor_to_seq(g, AMINO_ACIDS_MAP), "mask": m.tolist()}
                    self.outFile.write(json.dumps(d) + "\n")

    def remask(self, x, **kwargs) -> None:
        raise NotImplementedError

    def pairs(self, x, output) -> None:
        raise NotImplementedError


class SampleGeneratorStrokach(SampleGenerator):

    def __init__(self, outPath, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(outPath, repeats, 1, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)

    def pairs(self, x, output) -> None:
        return [(x.seq, output, x.mask)]


class SampleGeneratorJLo(SampleGenerator):

    def __init__(self, outPath, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(outPath, repeats, 1, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)
        x.maskedSeq = self.tokenizer.AA2BERT(x.maskedSeq)[0]

    def pairs(self, x, output) -> None:
        return [(x.seq, output, x.mask)]


class SampleGeneratorIngrham(SampleGenerator):

    def __init__(self, outPath, repeats, device = "cpu") -> None:
        super().__init__(outPath, repeats, -1, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        for i in range(x.seq.shape[0]):
            l = x.lengths[i]
            x.maskedSeq[i, :l], x.mask[i, :l] = maskBERT(x.seq[i, :l], **kwargs)

    def pairs(self, x, output) -> None:
        for i, l in enumerate(x.lengths):
            yield x.seq[i, :l], output[i, :l], x.mask[i, :l]


class SampleGeneratorIngrhamBERT(SampleGenerator):

    def __init__(self, outPath, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(outPath, repeats, -1, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        for i in range(x.seq.shape[0]):
            l = x.lengths[i]
            x.maskedSeq[i, :l], x.mask[i, :l] = maskBERT(x.seq[i, :l], **kwargs)
            x.maskedSeq[i, :l] = self.tokenizer.AA2BERT(x.maskedSeq[i, :l])

    def pairs(self, x, output) -> None:
        for i, l in enumerate(x.lengths):
            yield x.seq[i, :l], output[i, :l], x.mask[i, :l]
