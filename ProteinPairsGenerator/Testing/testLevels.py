from tqdm import tqdm
from pathlib import Path
from typing import Optional, List
import json

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
        device = "cpu",
        kAccuracy : List[int] = [1, 3, 5],
        store_history = False,
        out_path : Optional[Path] = None
    ) -> None:
        self.levels = levels
        self.repeats = repeats
        self.device = device
        self.kAccuracy = kAccuracy

        # Set up storing of history
        self.store_history = store_history
        
        if self.store_history:
            self.history = []
            assert isinstance(out_path, Path)
            self.out_path = out_path

    def run(self, model, dm, verbose = False, name = False, extra_out = {}) -> None:

        model.to(device=self.device)

        for level in self.levels:
            stepResults = []
            for _ in range(self.repeats):
                for x in tqdm(dm.test_dataloader()) if verbose else dm.test_dataloader():

                    self.remask(x, **level)

                    x = dm.transfer_batch_to_device(x, self.device)

                    self.x = x
                    res = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in model.step(x, extra_out).items()}

                    stepResults.append(res)

                    del x, res

            out = self.postprocess(stepResults)
            if name:
                out.update({"name": name})
            
            self.prettyPrint(level, out)

            if self.store_history:
                self.history.append(out)

            del stepResults, out

    def analyzeCorrector(self, x, output, corrector):
        raise NotImplementedError


    def postprocess(self, stepResults) -> None:
        
        nTotal = 0
        nCorrect = [0 for _ in self.kAccuracy]
        loss = 0
        blosum_score = 0
        confusion_matrix = torch.zeros(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        for step in stepResults:
            nTotal += step["nTotal"]
            loss += step["loss"]

            for i, k in enumerate(self.kAccuracy):
                nCorrect[i] += step["nCorrect_{}".format(k)]

            blosum_score += step["blosum"]
            confusion_matrix += step["confusion_matrix"]

        out = {"Loss": loss / len(stepResults)}
        for i, k in enumerate(self.kAccuracy):
                out.update({"Accuracy_{}".format(k): nCorrect[i] / nTotal})
        
        out.update({"BLOSUM SCORE": blosum_score / nTotal})
        out.update({"Confusion Matrix": confusion_matrix})

        return out


    def prettyPrint(self, level, results):
        print("-" * 20)
        print(", ".join([k + " = " + str(v) for k, v in level.items()]))
        print(*[k + " = "  + str(v) for k, v in results.items()], sep="\n")
        print("-" * 20)

    def remask(self, x, **kwargs) -> None:
        raise NotImplementedError

    def save_history(self):
        with open(self.out_path, "a") as f:
            json.dump(self.history, f)


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

        for i, l in enumerate(x.lengths):
            yTrue = x.seq[i, :l]
            corrOut = corrector(output[i, :l])[0]
            # yPred = output[i]
            yPred = torch.argmax(corrOut.data, -1)
            nCorrect += ((yPred == yTrue) * x.mask[i, :l]).sum()
        
        return {"nCorrectCorrector": nCorrect.item()}


class TestProteinDesignJLo(TestProteinDesign):

    def __init__( 
        self,
        levels,
        repeats,
        device = "cpu",
        kAccuracy : List[int] = [1, 3, 5],
        store_history = False,
        out_path : Optional[Path] = None) -> None:
        self.tokenizer = AdaptedTAPETokenizer()
        super().__init__(levels, repeats, device, kAccuracy, store_history, out_path)

    def remask(self, x, **kwargs) -> None:
        if not "substitutionMatrix" in kwargs:
            kwargs["substitutionMatrix"] = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))

        x.maskedSeq, x.mask = maskBERT(x.seq, **kwargs)
        x.maskedSeq = self.tokenizer.AA2BERT(x.maskedSeq)[0]
