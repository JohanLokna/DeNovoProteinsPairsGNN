import sys
from numpy.core.numeric import full
from sklearn.metrics import confusion_matrix 
import numpy as np
import json
import tqdm
from pathlib import Path

import torch

from ProteinPairsGenerator.PostProcessing import GuessDataModule, CorrectorFullSoftBERT
from ProteinPairsGenerator.BERTModel import TAPEWrapper
from ProteinPairsGenerator.utils import ScoreBLOSUM, AMINO_ACIDS_BASE

k_values = [1, 3, 5]
def get_acc_k(k, yTrue, yHat, yPred):
    yPred_k = torch.topk(yPred.data, k, -1).indices
    nCorrect = (torch.any(yPred_k.unsqueeze(-1) == yTrue, dim=-1)).sum()
    return nCorrect

device = "cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")


corrector = CorrectorFullSoftBERT.load_from_checkpoint("/mnt/ds3lab-scratch/jlokna/SoftCorrectorV2/Experiments/hidden_size=256_N=2_alpha=0.8_lr=0.0001_dropout=0.3/Checkpoints/last.ckpt", hidden_size=256, N=2, dropout=0.3, alpha=0.8, lr=1e-4)
corrector.to(device=device)

# Test Seq2Seq
blosum_scorer = ScoreBLOSUM()
blosum_scorer.to(device)

# Setup output

extra_out = {
  "blosum": lambda yTrue, yHat, yPred: blosum_scorer(yTrue, yHat),
  "confusion_matrix": lambda yTrue, yHat, yPred, mask=None : \
      confusion_matrix(yTrue.flatten().cpu().numpy(), 
                       yPred.flatten().cpu().numpy(), 
                       labels=np.arange(len(AMINO_ACIDS_BASE))),
  "nCorrect_1": lambda yTrue, yHat, yPred: get_acc_k(1, yTrue, yHat, yPred),
  "nCorrect_3": lambda yTrue, yHat, yPred: get_acc_k(3, yTrue, yHat, yPred),
  "nCorrect_5": lambda yTrue, yHat, yPred: get_acc_k(5, yTrue, yHat, yPred)
}

# Run tests
out = {"dir": {}, "rec": {}}
pathlist = Path(sys.argv[2] if len(sys.argv) > 2 else "data").glob('**/*')
for path in pathlist:

    # Accumulator 
    accum = {
      "nCorrect": [], 
      "nTotal": [], 
      "R2R": [], 
      "R2W": [], 
      "W2R":[], 
      "W2W":[], 
      "blosum":[],
      "confusion_matrix": [],
      "nCorrect_1": [],
      "nCorrect_3": [],
      "nCorrect_5": []
    }

    # Iterate over dataset
    dm = GuessDataModule(testSet=str(path))
    for x in tqdm.tqdm(dm.test_dataloader()):

        # Test datapoint
        x = dm.transfer_batch_to_device(x, device)
        res = corrector.step(x, extra_out)

        # Process results
        for key in accum.keys():
            value = res[key]

            if isinstance(res[key], torch.Tensor):
                accum[key].append(value.item())
            elif isinstance(res[key], np.ndarray):
                accum[key].append(value.tolist())
            else:
                accum[key].append(value)

    # Save accum
    full_name = path.stem
    design_style = full_name.split("_")[-1]
    model_name = "_".join(full_name.split("_")[:-1])
    out[design_style][model_name] = accum

# Save results
outfile = "results.json"
json.dump(out, open(outfile, "w+"))
