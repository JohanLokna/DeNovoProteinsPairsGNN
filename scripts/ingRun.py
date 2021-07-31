import numpy as np
import json, time
from tqdm import tqdm

import torch

from ProteinPairsGenerator.utils import AMINO_ACIDS_BASE, AMINO_ACID_NULL
from ProteinPairsGenerator.IngrahamV2 import IngrahamV2Model
from ProteinPairsGenerator.BERTModel import maskBERT

class StructureDataset():
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWY'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
        }

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                # Convert raw coords to np arrays
                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            print('Discarded', discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True, collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
            else:
                clusters.append(batch)
                batch = []
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch

def featurize(batch, device, useBERT : bool = False):
    """ Pack and pad batch into torch tensors """
    alphabet = ''.join(AMINO_ACIDS_BASE + [AMINO_ACID_NULL])
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    STrue = np.zeros([B, L_max], dtype=np.int32)
    SMasked = np.zeros([B, L_max], dtype=np.int32)

    nTokens = len(alphabet)
    mask = torch.zeros(B, L_max, dtype=torch.float32, device=device)
    maskLoss = torch.zeros(B, L_max, dtype=torch.float32, device=device)

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        STrue[i, :l] = indices

        if np.any(indices == 20):
            print(b["seq"])
            print(alphabet)

        mask[i, :l] = 1.0
        if useBERT:
            indicesBert, maskWithBert = maskBERT(torch.from_numpy(indices).to(dtype=torch.long), torch.ones(nTokens, nTokens))
            maskLoss[i, :l] = maskWithBert.to(device=device)
            indices = indicesBert.detach().cpu().numpy()
        else:
            maskLoss[i, :l] = 1.0

        SMasked[i, :l] = indices

    # Mask
    isnan = np.isnan(X)
    X[isnan] = 0.

    # Conversion
    STrue = torch.from_numpy(STrue).to(dtype=torch.long,device=device)
    SMasked = torch.from_numpy(SMasked).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    return X, STrue, SMasked, mask, maskLoss, lengths

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


device = "cuda:0"
model = IngrahamV2Model(20, 128, 128, 128, 3, 3, 21, 30, "full", 0, 0.1, True, False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.NLLLoss(reduction='none')

# Load the dataset
batch_tokens = 10000
datasets = [StructureDataset(file, truncate=None, max_length=2000) for file in ["training_100.jsonl", "validation.jsonl"]]
loader_train, loader_validation = [StructureLoader(d, batch_size=batch_tokens) for d in datasets]
print('Training:{}, Validation:{}'.format(*(len(d) for d in datasets)))


epochs = 100
for e in range(epochs):
    # Training epoch
    model.train()
    train_sum, train_weights = 0., 0.
    for train_i, batch in enumerate(tqdm(loader_train)):

        # Get a batch
        X, STrue, SMasked, mask, maskLoss, lengths = featurize(batch, device, useBERT=True)

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
        for _, batch in enumerate(loader_validation):
            X, STrue, SMasked, mask, maskLoss, lengths = featurize(batch, device)
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
