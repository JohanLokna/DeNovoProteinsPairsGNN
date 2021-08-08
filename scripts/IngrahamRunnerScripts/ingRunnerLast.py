import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

import torch

from ProteinPairsGenerator.IngrahamModel import StructureDataset, StructureLoader
from ProteinPairsGenerator.Testing import TestProteinDesignIngrham
from ProteinPairsGenerator.IngrahamModel import IngrahamDataModule, IngrahamModel

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
model = IngrahamModel(20, 128, 128, 128, 3, 3, 21, 30, "full", 0, 0.1, True, False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Load the dataset
batch_tokens = 10000
datasets = [StructureDataset(file, truncate=None, max_length=2000) for file in ["../../training_100.jsonl", "../../validation.jsonl"]]
loader_train, loader_validation = [StructureLoader(d, batch_size=batch_tokens) for d in datasets]


# Load the dataset
root = Path("../../../proteinNetTesting")
testing = [root.joinpath("processed/testing")]
dm = IngrahamDataModule(root, trainSet=testing, valSet=testing, testSet=testing) # Use small set for efficency
testerIngraham = TestProteinDesignIngrham([{"maskFrac": 0.15}, {"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 40, device)


# Set up saving
bestValAcc = 0
bestModelPath = Path("bestModel.ckpt")
lastModelPath = Path("lastModel.ckpt")

epochs = 100
for e in range(epochs):

    print("----------------------------\nIteraion {}\n----------------------------\n".format(e))

    # Training epoch
    model.train()
    train_sum, train_weights = 0., 0.
    for train_i, x in enumerate(tqdm(loader_train)):

        # Get a batch
        x = transfer_batch_to_device(x, device)
        X, STrue, SMasked, mask, maskLoss, lengths = x.coords, x.seq, x.maskedSeq, x.valid, x.mask, x.lengths

        optimizer.zero_grad()
        log_probs = model(X, SMasked, lengths, mask)
        _, loss_av_smoothed = loss_smoothed(STrue, log_probs, maskLoss, weight=0.1)
        loss_av_smoothed.backward()
        optimizer.step()

        loss, loss_av = loss_nll(STrue, log_probs, maskLoss)

        # Accumulate true loss
        train_sum += torch.sum(loss * maskLoss).cpu().data.numpy()
        train_weights += torch.sum(maskLoss).cpu().data.numpy()
    
    # Validation epoch
    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights, validation_correct = 0., 0., 0
        for _, x in enumerate(loader_validation):
            
            x = transfer_batch_to_device(x, device)
            X, STrue, SMasked, mask, maskLoss, lengths = x.coords, x.seq, x.maskedSeq, x.valid, x.mask, x.lengths
            
            log_probs = model(X, SMasked, lengths, mask)
            loss, loss_av = loss_nll(STrue, log_probs, maskLoss)

            # Accumulate
            validation_sum += torch.sum(loss * maskLoss).cpu().data.numpy()
            validation_correct += torch.sum((torch.argmax(log_probs.detach(), dim=-1) == STrue.detach()) * maskLoss)
            validation_weights += torch.sum(maskLoss).cpu().data.numpy()

    train_loss = train_sum / train_weights
    train_perplexity = np.exp(train_loss)
    validation_loss = validation_sum / validation_weights
    validation_perplexity = np.exp(validation_loss)
    validation_accuracy = validation_correct / validation_weights
    print('Perplexity\tTrain:{0: >#16.14f}\t\tValidation:{1: >#16.14f}'.format(train_perplexity, validation_perplexity))
    print('Accuracy\t\t\t\t\tValidation:{}'.format(validation_accuracy))

    # Save the best model
    if validation_accuracy > bestValAcc:
        bestValAcc = validation_accuracy
        bestModelPath.unlink(missing_ok=True)
        print("Saving model to ", str(bestModelPath))
        torch.save(model.state_dict(), str(bestModelPath))

    # Just to verify that everything is working
    if e % 10 == 0:
        testerIngraham.run(model, dm)

#Save the last model
torch.save(model.state_dict(), lastModelPath)
