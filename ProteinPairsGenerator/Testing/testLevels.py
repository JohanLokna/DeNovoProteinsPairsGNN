from tqdm import tqdm

import torch

# Local imports
from ProteinPairsGenerator.BERTModel import maskBERT
from ProteinPairsGenerator.utils import AMINO_ACIDS_BASE
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer

class TestProteinDesign:

    def __init__(
        self,
        dm,
        levels,
        repeats,
        device = "cpu"
    ) -> None:
        self.dm = dm
        self.levels = levels
        self.repeats = repeats
        self.device = device

    def run(self, model, verbose = False) -> None:

        model.to(device=self.device)

        for level in self.levels:
            stepResults = []
            for _ in range(self.repeats):
                for x in tqdm(self.dm.test_dataloader()) if verbose else self.dm.test_dataloader():

                    if torch.numel(x.seq) > 2500:
                        continue

                    print(x)

                    self.remask(x, **level)
                    x = self.dm.transfer_batch_to_device(x, self.device)

                    res = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.step(x).items()}

                    # stepResults.append(res)

                    # print(stepResults)

                    del x, res

            self.prettyPrint(level, self.postprocess(stepResults))

    def postprocess(self, stepResults) -> None:
        nTotal = 0
        nCorrect = 0
        loss = 0
        for step in stepResults:
            nTotal += step["nTotal"]
            nCorrect += step["nCorrect"]
            loss += step["loss"]
        return {"Accuracy": nCorrect / nTotal, "Loss": loss / len(stepResults)}

    def prettyPrint(self, level, results):
        print("-" * 20)
        print(", ".join([k + " = " + str(v) for k, v in level.items()]))
        print(*[k + " = "  + str(v) for k, v in results.items()], sep="\n")
        print("-" * 20)

    def remask(self, x, **kwargs) -> None:
        raise NotImplementedError


class TestProteinDesignStrokach(TestProteinDesign):

    def remask(self, x, **kwargs) -> None:
        assert(not "teacherLabels" in x.__dict__)
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)


class TestProteinDesignJLo(TestProteinDesign):

    def __init__(self, dm, levels, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(dm, levels, repeats, device)

    def remask(self, x, **kwargs) -> None:
        assert(not "teacherLabels" in x.__dict__)
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)
        x.maskedSeq = self.tokenizer.AA2BERT(x.maskedSeq)[0]

