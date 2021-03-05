from pathlib import Path
from prody import fetchPDB, pathPDBFolder, parsePDB, AtomGroup
from typing import List, Mapping, Tuple

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch_geometric.transforms as T


def base(data_pdb: AtomGroup) -> Data:
    
    # Get sequence
    seq = data_pdb.getSequence()

    # Find intersequence distance
    n = len(seq)

    ids = torch.arange(0, n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(ids)
    seq_distances = torch.cdist(ids, ids).flatten()

    print(seq_distances)

    # Compute caresian distances
    coords = torch.from_numpy(data_pdb.getCoords())
    cart_distances = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)

    # Mask and put in correct shape
    mask = (cart_distances < 12).flatten()
    edge_attr = torch.stack([cart_distances.flatten(), seq_distances.flatten()])

    print(edge_index.shape)

    edge_index = torch.stack(torch.where(mask))

    print(edge_attr.shape)

    # Create data point
    data = Data(x=seq, edge_index=edge_index, edge_attr=edge_attr)
    data = data.coalesce()

    # Assertions
    assert not data.contains_self_loops()
    assert data.is_coalesced()
    assert data.is_undirected()

    return data


class ProteinInMemoryDataset(InMemoryDataset):

    def __init__(
        self,
        root : Path,
        pdb_list : List[str],
        device : str = "cpu",
        transform = None,
        pre_transform : Mapping[AtomGroup, Data] = base,
        pre_filter = None,
    ) -> None:

        # Set up PDB
        pathPDBFolder(folder=root, divided=False)
        self.pdb_list_ = pdb_list

        transform = T.Compose(
            ([transform] if transform is not None else [])
        )
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [pdb + ".pdb" for pdb in self.pdb_list_]

    @property
    def processed_file_names(self):
        return [pdb + "_processed.pdb" for pdb in self.pdb_list_]

    def download(self):
        fetchPDB(self.pdb_list_, compressed=True)

    def process(self):
        data_list = [self.pre_transform(data_pdb) for data_pdb in parsePDB(self.pdb_list_)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), "test")
