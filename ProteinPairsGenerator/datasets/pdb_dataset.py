from pathlib import Path
from prody import fetchPDB, pathPDBFolder, parsePDB, AtomGroup
from typing import List, Mapping

import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T


class PDBInMemoryDataset(InMemoryDataset):

    def __init__(
        self,
        root : Path,
        pdb_list : List[str],
        pre_transform : Mapping[AtomGroup, Data],
        device : str = "cuda" if torch.cuda.is_available() else "cpu",
        transform_list : List[Mapping[Data, Data]] = [],
        pre_filter = None,
    ) -> None:

        # Set up PDB
        pathPDBFolder(folder=root, divided=False)
        self.pdb_list_ = pdb_list
        assert len(self.pdb_list_) > 0

        transform = T.Compose(transform_list)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):
        return [self.root.joinpath(pdb + ".pdb") for pdb in self.pdb_list_]

    @property
    def processed_file_names(self):
        return [self.root.joinpath("processed_pdb")]

    def download(self):
        fetchPDB(self.pdb_list_, compressed=True)

    def process(self):
        if len(self.pdb_list_) > 1:
          data_list = [self.pre_transform(data_pdb) for data_pdb in parsePDB(self.pdb_list_)]
        else:
          data_list = [self.pre_transform(parsePDB(self.pdb_list_))]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_names[0])
