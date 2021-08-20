import sys

import torch

from ProteinPairsGenerator.PostProcessing import GuessDataModule, CorrectorFullSoftBERT

device = "cuda:2" #"cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")

dm = GuessDataModule(testSet="guessesTestIngraham000.json")
corrector = CorrectorFullSoftBERT.load_from_checkpoint("/mnt/ds3lab-scratch/jlokna/SoftCorrectorV2/Experiments/hidden_size=256_N=2_alpha=0.8_lr=0.0001_dropout=0.3/Checkpoints/last.ckpt", hidden_size=256, N=2, dropout=0.3, alpha=0.8, lr=1e-4)
corrector.to(device=device)

accum = {"nCorrect": 0, "nTotal": 0, "R2R": 0, "R2W": 0, "W2R":0, "W2W":0}
for x in dm.test_dataloader():
    x = dm.transfer_batch_to_device(x, device)
    res = corrector.step(x)
    for k in accum.keys():
        accum[k] += (res[k].item() if isinstance(res[k], torch.Tensor) else res[k])
    print(accum)
