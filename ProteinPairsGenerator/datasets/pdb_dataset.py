from pathlib import Path
from prody import fetchPDB, pathPDBFolder, parsePDB, AtomGroup
from typing import Iterator, List, Mapping, Union

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch_geometric.transforms as T


def base(data: AtomGroup) -> (torch.Tensor, torch.Tensor):
    return torch.ones(1), torch.ones(1)


class ProteinInMemoryDataset(InMemoryDataset):

    def __init__(
        self,
        root : Path,
        pdb_list : List[str],
        device : str = "cpu",
        transform = None,
        pre_transform : Mapping[AtomGroup, (torch.Tensor, torch.Tensor)] = base,
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
        for data_pdb in parsePDB(self.pdb_list_):

            seq = data_pdb.getSequence()
            edge_index, edge_attr = self.edge_extracter(data_pdb)

        #     edge_index, edge_attr = remove_nans(edge_index, edge_attr)
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #     if data is not None:
        #         data_list.append(data)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])




