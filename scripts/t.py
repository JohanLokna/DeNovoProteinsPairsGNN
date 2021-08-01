from tqdm import tqdm
from pathlib import Path
import sys

import torch

from ProteinPairsGenerator.IngrahamModel import IngrahamModel, IngrahamDataModule, StructureDataset, StructureLoader
from ProteinPairsGenerator.Testing import TestProteinDesignIngrham

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, num_classes=20).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

device = "cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")
model = IngrahamModel(20, 128, 128, 128, 3, 3, 21, 30, "full", 0, 0.1, True, False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Load the dataset
batch_tokens = 10000
d = StructureDataset("training_100.jsonl", truncate=None, max_length=2000)
loader_train = StructureLoader(d, batch_size=batch_tokens)

# Testing
root = Path("../proteinNetNew")
testing = [Path("../proteinNetTesting").joinpath("processed/testing")]
dm = IngrahamDataModule(root, trainSet=testing, testSet=testing, teacher="TAPE")
testerIngraham = TestProteinDesignIngrham([{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 40, device)

epochs = 100
for e in range(epochs):

    print("----------------------------\nIteraion {}\n----------------------------\n".format(e))

    # Training epoch
    model.train()
    for batch in tqdm(loader_train):

        x = dm.transfer_batch_to_device(batch, device)

        optimizer.zero_grad()
        output = model(x.coords, x.maskedSeq, x.lengths, x.valid)
        _, loss = loss_smoothed(x.seq, output, x.mask)
        loss.backward()
        optimizer.step()

    # Validation epoch
    model.eval()
    with torch.no_grad():
        validation_weights, validation_correct = 0, 0
        for _, batch in enumerate(dm.val_dataloader()):
            
            outDict = model.step(dm.transfer_batch_to_device(batch, device))

            # Accumulate
            validation_correct += outDict["nCorrect"]
            validation_weights += outDict["nTotal"]

    print('Accuracy\t\t\t\t\tValidation:{}'.format(validation_correct / validation_weights))

    testerIngraham.run(model, dm)
