import prody
import sys

import torch

from ProteinPairsGenerator.utils.amino_acids import seq_to_tensor, AMINO_ACIDS_MAP

if __name__ == "__main__":
    pdb = prody.parsePDB(sys.argv[1]).ca

    # Get seqence
    seq = seq_to_tensor(pdb.getSequence(), mapping=AMINO_ACIDS_MAP)

    # Get Cartesian distance
    coords = torch.from_numpy(pdb.getCoordsets(0))
    dists = torch.cdist(coords, coords).squeeze(0)
    closeAA = dists < 12

    # Get sequence distance
    tmp = torch.arange(seq.shape[0].item()).view(-1, 1).expand(seq.shape[0].item(), 2)
    seqDists = tmp[:, 0] - tmp[:, 1].view(-1, 1)

    # Get edges
    edgeIdx = torch.stack(torch.where(closeAA), dim=0)
    edgeAtr = torch.stack([dists, seqDists], dim=-1).squeeze(dim=1)[closeAA.squeeze(dim=1)]

    #
    xBase = GeneralData(seq=seq, edge_index = edgeIdx, edge_attr = edgeAtr)
