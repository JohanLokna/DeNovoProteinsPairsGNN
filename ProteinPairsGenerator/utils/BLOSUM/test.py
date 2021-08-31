import pandas as pd

# Better to import from ../amino_acids but local package problems...
AMINO_ACIDS_BASE = [
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

# Transform matrix into csv
full_matrix = pd.read_csv(open("blosum62.txt", "r"), delimiter="\s*", index_col=0)
relevant_matrix = full_matrix[AMINO_ACIDS_BASE].loc[AMINO_ACIDS_BASE]
relevant_matrix.to_csv("blosum62.csv")
