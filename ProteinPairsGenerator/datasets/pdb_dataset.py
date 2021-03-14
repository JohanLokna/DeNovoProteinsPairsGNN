from pandas import DataFrame 
from pathlib import Path
from prody import fetchPDB, pathPDBFolder, parsePDB, AtomGroup
from typing import List, Mapping, Union

import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

def removeNone(x : Data) -> bool:
    return x is None


class PDBInMemoryDataset(InMemoryDataset):

    def __init__(
        self,
        root : Path,
        pdbs : Union[List[str], DataFrame],
        pre_transform : Mapping[AtomGroup, Data],
        device : str = "cuda" if torch.cuda.is_available() else "cpu",
        transform_list : List[Mapping[Data, Data]] = [],
        pre_filter : Mapping[Data, bool] = removeNone
    ) -> None:

        # Set up PDB
        pathPDBFolder(folder=root, divided=False)
        self.pdb_folder = root
        self.pdbs = pdbs
        assert len(self.pdbs) > 0

        transform = T.Compose(transform_list)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):
        if type(self.pdbs) is list:
            return [self.root.joinpath(pdb + ".pdb") for pdb in self.pdbs]
        else:
            return [self.root.joinpath(pdb + ".pdb") for pdb in self.pdbs.index.values.tolist()]

    @property
    def processed_file_names(self):
        return [self.root.joinpath("processed_pdb")]

    def download(self):
        print("Download")
        if type(self.pdbs) is list:
            fetchPDBviaHTTP(self.pdbs, compressed=True)
        else:
            fetchPDBviaHTTP(self.pdbs.index.values.tolist(), compressed=True)

    def process(self):
        print("Process")        
              
        if type(self.pdbs) is list:
            data_list = [self.pre_transform(data_pdb) for data_pdb in parsePDB(self.pdbs)]
        else:
            data_list = [self.pre_transform(parsePDB(pdb), **meta_data.to_dict()) \
                         for pdb, meta_data in self.pdbs.iterrows()]
        if not self.pre_filter is None:
            data_list = list(filter(self.pre_filter, data_list))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_names[0])
