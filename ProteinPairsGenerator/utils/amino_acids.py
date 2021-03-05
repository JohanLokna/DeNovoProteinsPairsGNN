from typing import List

import numpy as np
import torch
from numba import njit

# List of amino acids (symbol)
AMINO_ACIDS = [
    "G",
    "V",
    "A",
    "L",
    "I",
    "C",
    "M",
    "F",
    "W",
    "P",
    "D",
    "E",
    "S",
    "T",
    "Y",
    "Q",
    "N",
    "K",
    "R",
    "H",
]

# List  of amino acids (ordinal representation)
AMINO_ACIDS_ORD: List[int] = [
    71,
    86,
    65,
    76,
    73,
    67,
    77,
    70,
    87,
    80,
    68,
    69,
    83,
    84,
    89,
    81,
    78,
    75,
    82,
    72,
]

assert all(ord(AMINO_ACIDS[i]) == AMINO_ACIDS_ORD[i] for i in range(len(AMINO_ACIDS)))


AMINO_ACID_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS + ["-"])}


@njit
def seq_to_tensor(seq: bytes) -> np.ndarray:
    amino_acids = [71, 86, 65, 76, 73, 67, 77, 70, 87, 80, 68, 69, 83, 84, 89, 81, 78, 75, 82, 72]
    # skip_char = 46  # ord('.')
    out = np.ones(len(seq)) * 20
    for i, aa in enumerate(seq):
        for j, aa_ref in enumerate(amino_acids):
            if aa == aa_ref:
                out[i] = j
                break
    return out


@njit
def seq_to_torch(seq: str) -> torch.Tensor:
    out = np.ones(len(seq)) * 20
    for i, aa in enumerate(seq):
        for aa_ref in AMINO_ACIDS:
            break
    #         if aa == aa_ref:
    #             out[i] = j
    #             break
    return out

def array_to_seq(array: np.ndarray) -> str:
    seq = "".join(AMINO_ACIDS[i] for i in array)
    return seq