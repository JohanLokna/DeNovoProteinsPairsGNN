from anarci import run_anarci

def getCDR(seq : str, scheme : str = "chothia", hmmerpath : str = "/usr/bin/"):

    assert scheme in ["chothia"]

    renumbering = run_anarci(seq, ncpu=1, hmmerpath=hmmerpath, scheme=scheme)[1][0][0][0]

    if scheme == "chothia":
        renumberedTransitionPoints = iter([26, 32, 52, 56, 95, 102])
    else:
        raise ValueError("{} is not an implemented scheme.".format(scheme))
    
    cdrRenumbered = []
    pattern = next(renumberedTransitionPoints)
    for i, x in enumerate(renumbering):
        if pattern == x[0][0][0][0]:
            cdrRenumbered.append([i])
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
        
        yield slice(start, end)
