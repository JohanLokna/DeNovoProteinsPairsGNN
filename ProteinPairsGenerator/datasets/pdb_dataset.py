from pathlib import Path
from prody import fetchPDB, pathPDBFolder, parsePDB, AtomGroup
from typing import List, Mapping, Tuple

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

from ProteinPairsGenerator.utils import seq_to_torch
from .utils import transform_edge_attr

def base(data_pdb: AtomGroup) -> Data:

    set_pdb = data_pdb.select("name CA chain H")
    print(set_pdb.getSequence())
    
    # Get sequence
    seq = torch.tensor(
        seq_to_torch(set_pdb.getSequence()), dtype=torch.long
    )

    # Find intersequence distance
    n = seq.shape[0]

    ids = torch.arange(n, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
    seq_distances = torch.cdist(ids, ids).flatten()

    # Compute caresian distances
    coords = torch.from_numpy(set_pdb.getCoordsets())
    cart_distances = torch.cdist(coords, coords).squeeze(0)

    # Compute edges and their atributes
    mask = cart_distances < 12
    edge_attr = torch.stack(
      [cart_distances.flatten(), seq_distances.flatten()], dim=1
    )
    edge_attr = edge_attr[mask.flatten(), :]
    edge_index = torch.stack(torch.where(mask), dim=0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    # Create data point
    data = Data(x=seq, edge_index=edge_index, edge_attr=edge_attr)
    data = transform_edge_attr(data)
    data = data.coalesce()

    # Assertions
    assert not data.contains_self_loops()
    assert data.is_coalesced()

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
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root.joinpath(pdb + ".pdb") for pdb in self.pdb_list_]

    @property
    def processed_file_names(self):
        return [self.root.joinpath("processed_pdb")]

    def download(self):
        fetchPDB(self.pdb_list_, compressed=True)

    def process(self):
        data_list = [self.pre_transform(data_pdb) for data_pdb in parsePDB(self.pdb_list_)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_names[0])
