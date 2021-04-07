from typing import List

import numpy as np
import torch

# Chains
CHAIN_HEAVY : str = "H"
CHAIN_LIGHT : str = "L"
CHAIN_ANTIGEN : str = "A"
CHAIN_NULL : str = "X"
CHAINS: List[str] = [CHAIN_HEAVY, CHAIN_LIGHT, CHAIN_ANTIGEN, CHAIN_NULL]
CHAINS_ORD: List[int] = [ord(c) for c in CHAINS]

# Bijective mapping between numerical and symbolic representation
CHAINS_MAP = {c: i for i, c in enumerate(CHAINS)}
CHAINS_MAP.update({i: c for i, c in enumerate(CHAINS)})

# List of amino acids (symbol)
AMINO_ACID_NULL: str = "X"
CDRS_HEAVY = ["9", "8", "7"]
CDRS_LIGHT = ["1", "2", "3"]
AMINO_ACIDS_BASE: List[str] = [
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
AMINO_ACIDS = AMINO_ACIDS_BASE + [AMINO_ACID_NULL] + CDRS_HEAVY + CDRS_LIGHT

# List  of amino acids (ordinal representation)
AMINO_ACIDS_ORD: List[int] = [ord(aa) for aa in AMINO_ACIDS]

# Bijective mapping between numerical and symbolic representation
AMINO_ACIDS_MAP = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AMINO_ACIDS_MAP.update({i: aa for i, aa in enumerate(AMINO_ACIDS)})


def seq_to_tensor(seq : str) -> torch.Tensor:
    """Mapping between amino acid sequence representations"""
    return torch.as_tensor([AMINO_ACIDS_MAP[aa] for aa in seq]).type(torch.LongTensor)

def tensor_to_seq(seq : str) -> torch.Tensor:
    """Mapping between amino acid sequence representations"""
    return "".join([AMINO_ACIDS_MAP[int(aa.item())] for aa in seq])
