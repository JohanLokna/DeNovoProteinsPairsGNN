import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

import torch

from ProteinPairsGenerator.IngrahamModel import StructureDataset, StructureLoader
from ProteinPairsGenerator.Testing import TestProteinDesignIngrham
from ProteinPairsGenerator.IngrahamModel import IngrahamDataModule, IngrahamModelKD, IngrahamModel
from ProteinPairsGenerator.BERTModel import TAPEAnnotator

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def transfer_batch_to_device(x, device):
    x.__dict__.update((k, v.to(device=device)) for k, v in x.__dict__.items() if isinstance(v, torch.Tensor))
    return x


device = "cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

# Set up model dependent on alpha
if alpha != 0:
    model = IngrahamModelKD(alpha, 20, 128, 128, 128, 3, 3, 21, 30, "full", 0, 0.1, True, False).to(device)
    annotator = TAPEAnnotator(device=device)
    print("Training with KD Model for alpha = {}\n".format(str(alpha)))
else:
    model = IngrahamModel(20, 128, 128, 128, 3, 3, 21, 30, "full", 0, 0.1, True, False).to(device)
    annotator = lambda x, y: None
    print("Training with standard model\n")

optimizer = model.configure_optimizers()["optimizer"]

# Load the dataset
batch_tokens = 5000
datasets = [StructureDataset(file, truncate=None, max_length=2000) for file in ["../../training_100.jsonl", "../../validation.jsonl"]]
loader_train, loader_validation = [StructureLoader(d, batch_size=batch_tokens) for d in datasets]


# Load the dataset
root = Path("../../../proteinNetTesting")
testing = [root.joinpath("processed/testing")]
dm = IngrahamDataModule(root, trainSet=testing, valSet=testing, testSet=testing) # Use small set for efficency


# Set up saving
bestValAcc = 0
bestModelPath = Path("bestModel.ckpt")
lastModelPath = Path("lastModel.ckpt")

epochs = 100
for e in range(epochs):

    print("----------------------------\nIteraion {}\n----------------------------\n".format(e))

    # Training epoch
    model.train()
    for train_i, x in enumerate(tqdm(loader_train)):

        # Get a batch
        x = transfer_batch_to_device(x, device)
        x.teacherLabels = annotator(x.maskedSeq, x.valid)

        # Make a step
        optimizer.zero_grad()
        outDict = model.step(x)
        outDict["loss"].backward()
        optimizer.step()

        del outDict

    # Validation epoch
    model.eval()
    with torch.no_grad():
        nCorrect, nTotal = 0, 0
        for _, x in enumerate(loader_validation):
            
          # Get a batch
          x = transfer_batch_to_device(x, device)
          x.teacherLabels = annotator(x.maskedSeq, x.valid)

          # Make a step
          outDict = model.step(x)

          # Accumulate
          nCorrect += outDict["nCorrect"]
          nTotal += outDict["nTotal"]

          del outDict

    vallAcc = nCorrect / nTotal
    print('Accuracy\t\t\t\t\tValidation:{}'.format(vallAcc))

    # Save the best model
    if vallAcc > bestValAcc:
        bestValAcc = vallAcc
        bestModelPath.unlink(missing_ok=True)
        print("Saving model to ", str(bestModelPath))
        torch.save(model.state_dict(), str(bestModelPath))

    del nCorrect, nTotal

    # Just to verify that everything is working
    # if e % 10 == 0:
    #     testerIngraham = TestProteinDesignIngrham([{"maskFrac": 0.15}, {"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 40, device)
    #     testerIngraham.run(model, dm, addRandomKD = True)
    #     del testerIngraham

#Save the last model
torch.save(model.state_dict(), lastModelPath)
testerIngraham = TestProteinDesignIngrham([{"maskFrac": 0.15}, {"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 40, device)
testerIngraham.run(model, dm, addRandomKD = True)
