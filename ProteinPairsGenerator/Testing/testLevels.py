# Local imports
from ProteinPairsGenerator.BERTModel import maskBERT

class TestProteinDesign:

    def __init__(
        self,
        dm,
        levels,
        repeats,
    ) -> None:
        self.dm = dm
        self.levels = levels
        self.repeats = repeats

    def run(self, model) -> None:
        for level in self.levels:
            stepResults = []
            for _ in range(self.repeats):
                for _, x in self.dm.test_dataloader():
                    self.remask(x, **level)
                    stepResults.append(model.step(x))
            self.postprocess(stepResults)

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
        print(*[k + " = "  + str(v) for k, v in level.items()], sep="\n")
        print("-" * 20)


class TestProteinDesignStrokach(TestProteinDesign):

    def remask(self, x, *args, **kwargs) -> None:
        assert(self.teacher == None)
        x.maskedSeq, x.mask = maskBERT(x.seq, *args, **kwargs)

