import sys
from sklearn.metrics import confusion_matrix 
import numpy as np

import torch

from ProteinPairsGenerator.PostProcessing import GuessDataModule, CorrectorFullSoftBERT
from ProteinPairsGenerator.BERTModel import TAPEWrapper
from ProteinPairsGenerator.utils import ScoreBLOSUM, AMINO_ACIDS_BASE

device = "cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")

dm = GuessDataModule(testSet=sys.argv[2] if len(sys.argv) > 2 else None)
corrector = CorrectorFullSoftBERT.load_from_checkpoint("/mnt/ds3lab-scratch/jlokna/SoftCorrectorV2/Experiments/hidden_size=256_N=2_alpha=0.8_lr=0.0001_dropout=0.3/Checkpoints/last.ckpt", hidden_size=256, N=2, dropout=0.3, alpha=0.8, lr=1e-4)
corrector.to(device=device)

# Test Seq2Seq
blosum_scorer = ScoreBLOSUM()
blosum_scorer.to(device)

extra_out = {
  "blosum": lambda yTrue, yPred : blosum_scorer(yTrue, yPred),
  "confusion_matrix": lambda yTrue, yPred, mask=None : \
      confusion_matrix(yTrue.flatten().cpu().numpy(), 
                       yPred.flatten().cpu().numpy(), 
                       labels=np.arange(len(AMINO_ACIDS_BASE)))
}

accum = {"nCorrect": 0, "nTotal": 0, "R2R": 0, "R2W": 0, "W2R":0, "W2W":0, "blosum":0}
acc = 0
cm_base = np.zeros((len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE)))
cm_corr = np.zeros((len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE)))

for x in dm.test_dataloader():
    x = dm.transfer_batch_to_device(x, device)
    res = corrector.step(x, extra_out)
    for k in accum.keys():
        accum[k] += (res[k].item() if isinstance(res[k], torch.Tensor) else res[k])
    
    acc += res["nCorrect"] / res["nTotal"]
    cm_corr += res["confusion_matrix"]
    cm_base += confusion_matrix(x.y.cpu().numpy(), x.x.cpu().numpy(), labels=np.arange(len(AMINO_ACIDS_BASE)))

accum.update({"avg_acc": acc / len(dm.test_dataloader()), 
              "confusion_matrix_corr": cm_corr.tolist(), 
              "confusion_matrix_base": cm_base.tolist()})
print(accum)
