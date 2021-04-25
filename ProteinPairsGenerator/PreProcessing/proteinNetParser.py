# General imports
from typing import Dict

# PyTorch imports
import torch

# Local Imports
from ProteinPairsGenerator.utils import AMINO_ACIDS_MAP, AMINO_ACIDS_BASE, seq_to_tensor

# Constants
NUM_DIMENSIONS = 3

def readPotein(inFile) -> Dict:
    """Read ProteinNet record. Based on implementation in their code exaples."""
    
    record = {}
    while True:
        line = inFile.readline()
        
        if "[ID]" in line:
            record["id"] = inFile.readline()[:-1]
        elif "[PRIMARY]" in line:
            record["primary"] = seq_to_tensor(inFile.readline()[:-1], mapping=AMINO_ACIDS_MAP)
        elif "[EVOLUTIONARY]" in line:
            evolutionary = torch.Tensor([
              [step for step in inFile.readline().split()] for residue in range(len(AMINO_ACIDS_BASE) + 1)
            ])
            record["evolutionary"] = evolutionary
        elif "[SECONDARY]" in line:
            # Not yet implemented - just skipping lines
            inFile.readline()
        elif "[TERTIARY]" in line:
            coords = torch.Tensor([
              [step for step in inFile.readline().split()] for residue in range(NUM_DIMENSIONS)
            ])
            record["N"] = coords[:, 0::3]
            record["CA"] = coords[:, 1::3]
            record["C"] = coords[:, 2::3]
        elif "[MASK]" in line:
            # Not yet implemented - just skipping lines
            inFile.readline()
        elif "\n" in line:
            return record
        else:
            return None
