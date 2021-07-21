# Local import
from ProteinPairsGenerator.utils import setupRun, runBayesianHP

def wrapperStrokach(kwargs):

    if "hidden_size" in kwargs:
        kwargs["hidden_size"] = int(kwargs["hidden_size"]) # int(2 ** kwargs["hidden_size"])
    if "N" in kwargs:
        kwargs["N"] = int(kwargs["N"])
    if "lr" in kwargs:
        kwargs["lr"] = 10 ** kwargs["lr"]

    print("Running new experiment", kwargs, sep="\n", end="\n\n")

    return setupRun(**kwargs)


if __name__ == "__main__":

    # Bounded region of parameter space
    pbounds = {"hidden_size": (7, 9), "N": (2.5, 4.5), "alpha": (0, 0.1), "lr": (-5, -3)}
    fixedPoints = [({"hidden_size": 128, "N": 3, "alpha": 0, "lr": -4, "alpha":0},),
                   ({"hidden_size": 128, "N": 3, "alpha": 0, "lr": -4, "alpha":0.5},),
                   ({"hidden_size": 128, "N": 3, "alpha": 0, "lr": -4, "alpha": 0.75},),
                   ({"hidden_size": 128, "N": 3, "alpha": 0, "lr": -4, "alpha": 0.25},),
                   ({"hidden_size": 128, "N": 3, "alpha": 0, "lr": -4, "alpha": 0.875},),
                   ({"hidden_size": 128, "N": 3, "alpha": 0, "lr": -4, "alpha": 0.625},),]
    runBayesianHP(pbounds, wrapperStrokach, nIter = len(fixedPoints), nParalell = 1, kind="ei", xi=4, kappa = 400, fixedPoints=fixedPoints)
