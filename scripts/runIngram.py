import numpy as np
import torch
from tqdm import tqdm

# Library code
from ProteinPairsGenerator.IngrahamModel import IngrahamDataModule
from ProteinPairsGenerator.IngrahamV2 import IngrahamV2Model

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed(S, log_probs, mask, weight=0.1, vocab_size = 20):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, num_classes = vocab_size).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

device = "cuda:0"
model = IngrahamV2Model(20, 128, 128, 128, 3, 3, 21, 30, "full", 0, 0.1, True, False).to(device)
optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.NLLLoss(reduction='none')

dm = IngrahamDataModule("proteinNetNew", batch_size = 16)
loader_train, loader_validation = dm.train_dataloader(), dm.val_dataloader()

epochs = 100
for e in range(epochs):
    
    # Training epoch
    model.train()
    train_sum, train_weights = 0., 0.
    print("Epoch {} / {}".format(e + 1, epochs))
    
    for x in tqdm(loader_train, total=len(loader_train)):

        x = dm.transfer_batch_to_device(x, device)

        optimizer.zero_grad()
        log_probs = model(x.coords, x.maskedSeq, x.lengths, x.valid)
        _, loss_av_smoothed = loss_smoothed(x.seq, log_probs, x.mask, weight=0.1)
        loss_av_smoothed.backward()
        optimizer.step()

        loss, loss_av = loss_nll(x.seq, log_probs, x.mask)

        # Accumulate true loss
        train_sum += torch.sum(loss * x.mask).cpu().data.numpy()
        train_weights += torch.sum(x.mask).cpu().data.numpy()

    # Validation epoch
    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights, validation_correct = 0., 0., 0
        for _, batch in enumerate(loader_validation):

            x = dm.transfer_batch_to_device(x, device)

            log_probs = model(x.coords, x.maskedSeq, x.lengths, x.valid)
            loss, loss_av = loss_nll(x.seq, log_probs, x.mask)

            # Accumulate
            validation_sum += torch.sum(loss * x.mask).cpu().data.numpy()
            validation_correct += torch.sum((torch.argmax(log_probs.detach(), dim=-1) == x.seq.detach()) * x.mask)
            validation_weights += torch.sum(x.mask).cpu().data.numpy()

    train_loss = train_sum / train_weights
    train_perplexity = np.exp(train_loss)
    validation_loss = validation_sum / validation_weights
    validation_perplexity = np.exp(validation_loss)
    validation_accuracy = validation_correct / validation_weights
    print('Perplexity\tTrain:{0: >#16.14f}\t\tValidation:{1: >#16.14f}'.format(train_perplexity, validation_perplexity))
    print('Accuracy\t\t\t\t\tValidation:{}'.format(validation_accuracy))
