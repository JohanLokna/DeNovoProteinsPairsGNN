from ProteinPairsGenerator.utils import tensor_to_seq
from anarci import run_anarci

def getCDR(seq : str, scheme : str = "chothia", hmmerpath : str = "/usr/bin/"):

    assert scheme in ["chothia"]

    renumbering = run_anarci(seq, ncpu=1, hmmerpath="/usr/bin/", scheme=scheme)[1][0][0][0]

    renumbered_points = iter([26, 32, 52, 56, 95, 102])
    seq_points = []
    
    pattern = next(renumbered_points)
    for i, x in enumerate(renumbering):
        if pattern == x[0][0][0][0]:
            seq_points.append(i)
            try:
                pattern = next(renumbered_points)
            except StopIteration:
                break

    cdr_points = iter(seq_points)
    while True:
        try:
            start = next(cdr_points)
            end = next(cdr_points)
        except StopIteration:
            break
        
        yield slice(start, end)
