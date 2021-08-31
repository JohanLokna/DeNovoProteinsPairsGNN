import numpy as np
import json, time

import torch

from ProteinPairsGenerator.BERTModel import maskBERT
from ProteinPairsGenerator.Data import GeneralData
from ProteinPairsGenerator.utils import seq_to_tensor, AMINO_ACIDS_MAP, AMINO_ACIDS_BASE

"""
    Helper function extracting the features for the Ingraham model
"""
def featurize(batch):

    # Set up data
    B = len(batch)
    lengths = [len(b["seq"]) for b in batch]
    L_max = max(lengths)
    coords = torch.zeros(B, L_max, 4, 3, dtype=torch.float)
    seq = torch.zeros(B, L_max, dtype=torch.long)
    maskedSeq = torch.zeros(B, L_max, dtype=torch.long)
    mask = torch.zeros(B, L_max)
    valid = torch.zeros(B, L_max)

    
    # Helpers
    nTokens = len(AMINO_ACIDS_BASE)

    # Build the batch
    for i, (b, l) in enumerate(zip(batch, lengths)):
        # Standard features
        numpyCorrds = np.nan_to_num(np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1))
        coords[i, :l] = torch.from_numpy(numpyCorrds).float()
        seq[i, :l] = seq_to_tensor(b["seq"], AMINO_ACIDS_MAP)
        valid[i, :l] = 1.0

        # Randomly masked sequence
        maskedSeq[i, :l], mask[i, :l] = maskBERT(seq[i, :l].detach(), torch.ones(nTokens, nTokens))

    return GeneralData(
        coords=coords,
        seq=seq,
        valid=valid,
        lengths=lengths,
        maskedSeq=maskedSeq,
        mask=mask
    )

"""
    Dataset specially used by Ingraham model data
    I am afraid my attempts of fitting the dataset into the format given in the data section 
    might have had a slight bug. This format, which was given in the paper, proved much better.
    When I tried to test, I found that both datasets contained the same information.
    It did not seem fruitful to try to debug the other format anymore when this worked;
    I gave up on that project after having struggled quite a few couple days.
"""
class StructureDataset():
    def __init__(self, jsonl_file, max_length, verbose=False, truncate=None):
        
        alphabet = ''.join(AMINO_ACIDS_BASE)
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

            if verbose:
                print('Discarded', discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


"""
    Loader used by Ingraham model
"""
class StructureLoader():
    def __init__(self, dataset : StructureDataset, batch_size : int, shuffle : bool = True):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.alphabet = set(AMINO_ACIDS_BASE)

        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:

            if not set(dataset[ix]['seq']) <= self.alphabet:
                continue

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
        
        if self.shuffle:
            np.random.shuffle(self.clusters)
        
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            batch = featurize(batch)
            yield batch
