from anarci import run_anarci
from typing import List


def getLightCDR(seq : str, scheme : str = "chothia", hmmerpath : str = "/usr/bin/"):

    assert scheme in ["chothia"]

    if scheme == "chothia":
        renumberedTransitionPoints = iter([24, 34, 50, 56, 89, 97])
    else:
        raise ValueError("{} is not an implemented scheme.".format(scheme))
      
    return getCDR(seq, renumberedTransitionPoints, scheme, hmmerpath)


def getHeavyCDR(seq : str, scheme : str = "chothia", hmmerpath : str = "/usr/bin/"):

    assert scheme in ["chothia"]

    if scheme == "chothia":
        renumberedTransitionPoints = iter([26, 32, 52, 56, 95, 102])
    else:
        raise ValueError("{} is not an implemented scheme.".format(scheme))
      
    return getCDR(seq, renumberedTransitionPoints, scheme, hmmerpath)


def getCDR(seq : str, renumberedTransitionPoints : List[int], scheme : str = "chothia",  hmmerpath : str = "/usr/bin/"):

    renumbering = run_anarci(seq, ncpu=1, hmmerpath=hmmerpath, scheme=scheme)[1][0][0][0]
    
    cdrRenumbered = []
    pattern = next(renumberedTransitionPoints)
    for i, x in enumerate(list(renumbering)):
        if pattern == x[0][0]:
            cdrRenumbered.append(i)
            try:
                pattern = next(renumberedTransitionPoints)
            except StopIteration:
                break

    cdrSeq = iter(cdrRenumbered)
    while True:
        try:
            start = next(cdrSeq)
            end = next(cdrSeq)
        except StopIteration:
            break
        yield slice(start - 1, end)
