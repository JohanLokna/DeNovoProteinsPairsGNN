# Local import
from ProteinPairsGenerator.utils import setupRun, runBayesianHP

def wrapperStrokach(kwargs):
    if "hidden_size" in kwargs:
        kwargs["hidden_size"] = int(2 ** kwargs["hidden_size"])
    if "N" in kwargs:
        kwargs["N"] = int(kwargs["N"])
    if "lr" in kwargs:
        kwargs["lr"] = 10 ** kwargs["lr"]

    print("Running new experiment", kwargs, sep="\n", end="\n\n")

    return setupRun(**kwargs)


if __name__ == "__main__":

    # Bounded region of parameter space
    pbounds = {"hidden_size": (5, 9), "N": (2, 4), "alpha": (0, 1), "lr": (-6, -2)}
    runBayesianHP(pbounds, wrapperStrokach, nIter = 3, nParalell  = 2)
