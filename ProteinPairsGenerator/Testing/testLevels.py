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

    def run(self, model, dm, verbose = False, addRandomKD = False, corrector = None, classDim = None) -> None:

        model.to(device=self.device)

        if corrector:
            corrector.to(device=self.device)

            def wrapper(*args, **kwargs):
                baseModelOutput = type(model).forward(*args, **kwargs)
                self.output = torch.argmax(baseModelOutput.data, classDim)
                return baseModelOutput

            model.forward = wrapper.__get__(model, type(model))


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
                        res.update(self.analyzeCorrector(x, self.output, corrector))
                        # assert res["n"] == res["nTotal"]
                        print(res)

                    stepResults.append(res)

                    del x, res

            self.prettyPrint(level, self.postprocess(stepResults, not (corrector is None)))
            del stepResults

    def analyzeCorrector(self, x, output, corrector):
        raise NotImplementedError


    def postprocess(self, stepResults, useCorrector = False) -> None:
        
        nTotal = 0
        nCorrect = 0
        loss = 0
        nCorrectCorrector = 0
        n = 0

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

    def analyzeCorrector(self, x, output, corrector):

        nCorrect = 0

        for seq, out, l in zip(torch.unbind(x.seq), torch.unbind(output), x.lengths):
            
            yTrue = seq[:l]
            
            corrOut = corrector(out[:l])[0]
            yPred = torch.argmax(corrOut.data, -1)

            nCorrect = (yPred == yTrue).sum()
        
        return {"nCorrectCorrector": nCorrect.item(), "n": sum(x.lengths)}


class TestProteinDesignJLo(TestProteinDesign):

    def __init__(self, levels, repeats, device = "cpu") -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(levels, repeats, device)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)
        x.maskedSeq = self.tokenizer.AA2BERT(x.maskedSeq)[0]
