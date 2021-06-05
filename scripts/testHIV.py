import copy
import prody

import torch

from ProteinPairsGenerator.utils.amino_acids import seq_to_tensor, tensor_to_seq, AMINO_ACIDS_MAP
from ProteinPairsGenerator.JLoModel import JLoModel
from ProteinPairsGenerator.BERTModel import AdaptedTAPETokenizer

mutations = {
  "A": [
    [],
    [("Q", "R", [157, 197, 241])],
    [("Q", "W", [155, 195, 239])]
  ],  
  "B": [
    [],
    [("Q", "R", [156, 196, 240])],
    [("Q", "W", [154, 194, 238])]
  ],
  "C": [
    [],
    [("Q", "R", [155, 195, 239])],
    [("Q", "W", [153, 193, 237])]
  ]
}

designs = {
  "A": ("./HIV/bnD7_A_r2_0002.pdb", [9, 12, 13, 14, 15, 16, 35, 37, 45, 46, 47, 48, 66, 68, 70, 74, 75, 78, 79, 81, 99, 100, 101, 103, 108, 111, 112, 113, 114]),
  "B": ("./bnD7_B_r.pdb", [8, 11, 12, 13, 14, 15, 34, 36, 40, 41, 44, 45, 46, 47, 49, 50, 65, 67, 69, 73, 74, 77, 78, 79, 80, 98, 99, 100, 102, 106, 107, 110, 111, 112, 113]),
  "C": ("./HIV/bnD7_C_r3_0025.pdb", [7, 10, 11, 12, 31, 32, 33, 35, 43, 44, 45, 46, 64, 66, 68, 69, 72, 73, 76, 77, 79, 97, 98, 99, 100, 101, 102, 105, 106, 109, 110, 111, 112])
}

if __name__ == "__main__":

    
    # Get model
    model = JLoModel.load_from_checkpoint("ExpJLo0.0/Checkpoints/epoch=25-step=2588299.ckpt", 
                                          x_input_size=21, adj_input_size=2, hidden_size=128, output_size=20, N=3)

    for key, (path, testPos) in designs.items():

        pdb = prody.parsePDB(path).ca

        # Get seqence
        seqBase = seq_to_tensor(pdb.getSequence(), mapping=AMINO_ACIDS_MAP)

        # Get Cartesian distance
        coords = torch.from_numpy(pdb.getCoordsets(0))
        dists = torch.cdist(coords, coords).squeeze(0)
        closeAA = dists < 12

        # Get sequence distance
        tmp = torch.arange(seqBase.shape[0].item()).view(-1, 1).expand(seqBase.shape[0].item(), 2)
        seqDists = tmp[:, 0] - tmp[:, 1].view(-1, 1)

        # Get edges
        edgeIdx = torch.stack(torch.where(closeAA), dim=0)
        edgeAtr = torch.stack([dists, seqDists], dim=-1).squeeze(dim=1)[closeAA.squeeze(dim=1)]

        # Get base skeleton
        tokenizer = AdaptedTAPETokenizer()

        for m in mutations[key]:

            seq = copy.deepcopy(seqBase)
            
            # Add mutations
            for old, new, positions in m:
                assert torch.all(seq[positions] == old)
                seq[positions] = new
            
            seq[testPos] = 20

            seq = tokenizer.AA2BERT(seq)[0]
            output = model(seq, edgeIdx, edgeAtr)
            print(torch.argmax(output.data, 1), mapping=AMINO_ACIDS_MAP)

