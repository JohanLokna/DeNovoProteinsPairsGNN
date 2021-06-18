# Local import
from ProteinPairsGenerator.utils import setupRun, runBayesianHP

def testing(**kwargs):
    print(kwargs)
    return 1


if __name__ == "__main__":

    # Bounded region of parameter space
    pbounds = {"hidden_size": (5, 9), "N": (2, 4), "alpha": (0, 5), "lr": (-6, -2)}

    runBayesianHP(pbounds, testing)