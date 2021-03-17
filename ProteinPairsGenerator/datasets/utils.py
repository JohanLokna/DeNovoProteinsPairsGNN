import numpy as np
from sklearn.cluster import AgglomerativeClustering

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Mapping, Union


def normalize_cart_distances(cart_distances):
    return (cart_distances - 6) / 12


def normalize_seq_distances(seq_distances):
    return (seq_distances - 0) / 68.1319


def transform_edge_attr(data):
    cart_distances = data.edge_attr
    cart_distances = normalize_cart_distances(cart_distances)
    seq_distances = (data.edge_index[1] - data.edge_index[0]).to(torch.float).unsqueeze(1)
    seq_distances = normalize_seq_distances(seq_distances)
    data.edge_attr = torch.cat([cart_distances, seq_distances], dim=1)
    return data


def removeNone(x : Data) -> bool:
    return not x is None


class splitDistinctSequences:

    def __init__(
        self, 
        dist : Mapping[Union[Data, Data], float],
        threshold
    ) -> None:
        self.dist = dist
        self.threshold = threshold

    def __call__(self, dataset : Dataset, *sizes):

        n = len(dataset)
        X = torch.zeros(n, n)

        # To generate lower half
        for i, x in enumerate(dataset):
            for j, y in enumerate(dataset):
                X[i, j] = self.dist(x, y)
                if j == i:
                    X[i, j] *= 0.5
                    break
        
        # Only computed lower half,
        # so add upper half by addtion
        X += X.transpose(0, 1).clone().detach()

        # Cluster
        points = np.arange(n).reshape(n, 1)
        model = AgglomerativeClustering(affinity=lambda x: X, 
                                        distance_threshold=self.threshold, 
                                        n_clusters=None, 
                                        linkage="complete")
        clustering = model.fit(points)
        unique, counts = np.unique(clustering.labels_, return_counts=True)
        idx = np.argsort(counts, axis=0)
        unique = np.take_along_axis(unique, idx, axis=0)
        counts = np.take_along_axis(counts, idx, axis=0)

        cumsum = np.cumsum(counts)

        print(unique, unique.shape)
        print(counts, counts.shape)
        print(cumsum, cumsum.shape)

        curr = 0
        nxt = 0
        for s in sizes:

            # Compute upper bound
            lower_bound = cumsum[curr] + s * n
            print(np.argmax(cumsum[curr :] > lower_bound), np.argmax(cumsum[curr :] > lower_bound).shape)
            nxt += np.argmax(cumsum[curr :] > lower_bound) + 1
            assert nxt < n
            
            # Return idex set
            idx = np.where(np.isin(clustering.labels_, unique[curr:nxt]))[0]
            yield torch.from_numpy(idx)

            # Set next lower bound
            curr = nxt
 
        # Return the rest
        idx = np.where(np.isin(clustering.labels_, unique[curr:]))[0]
        yield torch.from_numpy(idx)
