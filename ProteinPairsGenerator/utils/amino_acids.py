from typing import List, Mapping

import numpy as np
import torch

# List of chains (symbol)
CHAINS : List[str] = [
  "L",
  "H",
  "B",
  "E",
  "G",
  "I",
  "T",
  "S"
]
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

def seq_to_tensor(seq : str, mapping : Mapping[str, int]) -> torch.Tensor:
    """Mapping between amino acid sequence representations"""
    return torch.as_tensor([mapping[aa] for aa in seq]).type(torch.LongTensor)

def tensor_to_seq(seq : torch.Tensor, mapping : Mapping[int, str]) -> str:
    """Mapping between amino acid sequence representations"""
    return "".join([mapping[int(aa.item())] for aa in seq])
