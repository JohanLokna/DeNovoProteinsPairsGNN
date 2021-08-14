from tqdm import tqdm

import torch

# Local imports
from ProteinPairsGenerator.BERTModel import maskBERT
from ProteinPairsGenerator.utils import AMINO_ACIDS_BASE
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer

class TestProteinDesign:

    def __init__(
        self,
        levels,
        repeats,
        device = "cpu"
    ) -> None:
        self.levels = levels
        self.repeats = repeats
        self.device = device

    def run(self, model, dm, verbose = False, addRandomKD = False, corrector = None) -> None:

        model.to(device=self.device)

        if corrector:
            corrector.to(device=self.device)

        for level in self.levels:
            stepResults = []
            for _ in range(self.repeats):
                for x in tqdm(dm.test_dataloader()) if verbose else dm.test_dataloader():

                    self.remask(x, **level)

                    if addRandomKD:
                        x.teacherLabels = torch.rand(*x.seq.shape, 20).type_as(x.coords)

                    x = dm.transfer_batch_to_device(x, self.device)

                    res = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in model.step(x).items()}

                    # Add corrector
                    if corrector:
                        outCorrector = self.analyzeCorrector(self.getSeq(x), corrector(x))
                        res.update(outCorrector)

                    stepResults.append(res)

                    print(res)

                    del x, res

            self.prettyPrint(level, self.postprocess(stepResults, not (corrector is None)))
            del stepResults

    def getSeq(self, x):
        return x.seq

    def analyzeCorrector(self, yTrue, output):
        yPred = torch.argmax(output.data, -1)
        nCorrect = (yPred == yTrue).sum()
        return {"nCorrectCorrector": nCorrect}


    def postprocess(self, stepResults, useCorrector = False) -> None:
        
        nTotal = 0
        nCorrect = 0
        loss = 0
        nCorrectCorrector = 0

        for step in stepResults:
            nTotal += step["nTotal"]
            nCorrect += step["nCorrect"]
            loss += step["loss"]

            if useCorrector:
                nCorrectCorrector += step["nCorrectCorrector"]

        out = {"Accuracy": nCorrect / nTotal, "Loss": loss / len(stepResults)}

        if useCorrector:
            out.update({"Accuracy Corrector": nCorrectCorrector / nTotal})
        
        return out


    def prettyPrint(self, level, results):
        print("-" * 20)
        print(", ".join([k + " = " + str(v) for k, v in level.items()]))
        print(*[k + " = "  + str(v) for k, v in results.items()], sep="\n")
        print("-" * 20)

    def remask(self, x, **kwargs) -> None:
        raise NotImplementedError


class TestProteinDesignStrokach(TestProteinDesign):

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)


class TestProteinDesignIngrham(TestProteinDesign):

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        for i in range(x.seq.shape[0]):
            l = x.lengths[i]
            x.maskedSeq[i, :l], x.mask[i, :l] = maskBERT(x.seq[i, :l], **kwargs)


class TestProteinDesignJLo(TestProteinDesign):

    def __init__(self, levels, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(levels, repeats, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)
        x.maskedSeq = self.tokenizer.AA2BERT(x.maskedSeq)[0]
