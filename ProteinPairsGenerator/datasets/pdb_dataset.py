from multiprocessing import Pool
import pandas as pd
from pathlib import Path
from prody import fetchPDBviaHTTP, pathPDBFolder, parsePDB, AtomGroup
from typing import List, Mapping, Callable, Union, Generator, Any

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

from .utils import removeNone
from .pdb_data import PDBData   
from .compute_modules import GetSequence, GetSequenceCDR, GetChainsDescription


def wrapper(func, **kwargs):
    return func(**kwargs)


class PDBBuilder:

    def __init__(
        self, 
        seqExtracter = None,
        xExtracter = None,
        yExtracter = None,
        edgeAttrExtracter = None,
        edgeFilter = None
    ) -> None:
        
        # Set up members
        self.seqExtracter = seqExtracter
        self.xExtracter = xExtracter
        self.yExtracter = yExtracter
        self.edgeAttrExtracter = edgeAttrExtracter
        self.edgeFilter = edgeFilter

    def __call__(
        self,
        pdb: AtomGroup,
        **kwargs
    ) -> PDBData:
        
        try:

            assert pdb.numAtoms() < 7000

            # Only consider alpha Cs
            pdbCAlpha = pdb.ca
            n = pdbCAlpha.numAtoms()

            # Extract sequence
            seq = self.seqExtracter(pdbCAlpha, **kwargs)

            # Extract y data
            y = self.yExtracter(pdbCAlpha, **kwargs)
            
            # Extract node features
            if not self.xExtracter is None:
                x = self.xExtracter(pdbCAlpha, **kwargs)
            else:
                x = None

            # Extract edge features
            if not self.edgeAttrExtracter is None:
                edgeAttr = self.edgeAttrExtracter(pdbCAlpha, **kwargs)

                # Ensure edge feautes have correct shape
                if len(edgeAttr.size()) == 2:
                    edge_attr = edge_attr.unsqueeze(-1)
                if len(edgeAttr.size()) != 3:
                    raise Exception

                # Find valid edges
                if not self.edgeFilter is None:
                    mask = self.edgeFilter(pdbCAlpha, edgeAttr, **kwargs)
                else:
                    mask = torch.ones(n, n, dtype=torch.bool)

                # Transform to sparse form
                edgeAttr = edgeAttr[mask, :].view(-1, edgeAttr.size()[-1])
                edgeIdx = torch.stack(torch.where(mask), dim=0).type(torch.LongTensor)
                edgeIdx, edgeAttr = remove_self_loops(edgeIdx, edgeAttr)

            else:
                edgeAttr = None
                edgeIdx = None

            meta = {'chains': GetChainsDescription()(pdbCAlpha, **kwargs)}

            # Assertions
            data = PDBData(seq=seq, x=x, edge_index=edgeIdx, edge_attr=edgeAttr, y=y, meta=meta)
            assert not data.contains_self_loops()
            assert data.is_coalesced()

            return data

        except Exception:
            return None


class PDBInMemoryDataset(InMemoryDataset):

    def __init__(
        self,
        root : Path,
        pdbs : Union[List[str], pd.DataFrame],
        pre_filter : Mapping[PDBData, bool],
        pre_transform : Mapping[AtomGroup, PDBData],
        device : str = "cuda" if torch.cuda.is_available() else "cpu",
        transform_list : List[Mapping[PDBData, PDBData]] = [],
        pdbFolders : List[Path] = [],
        pool: Union[Pool, None] = None
    ) -> None:

        # Set up root
        self.root = root
        self.root.mkdir(exist_ok=True)

        # Set up PDB
        self.pdbs = pdbs        
        pdbFolders.append(self.raw_dir)
        for folder in pdbFolders:
            folder.mkdir(exist_ok=True)
            pathPDBFolder(folder=folder, divided=False)

        # Set up pool
        self.pool = pool
        
        # Initialize super class and complete set up
        super().__init__(root, T.Compose(transform_list), pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self) -> List[Path]:
        if type(self.pdbs) is list:
            return [self.raw_dir.joinpath(pdb + ".pdb") for pdb in self.pdbs]
        elif type(self.pdbs) is pd.DataFrame:
            return [self.raw_dir.joinpath(pdb + ".pdb") for pdb in self.pdbs.index.values.tolist()]
        else:
            raise NotImplementedError

    @property
    def processed_file_names(self) -> List[Path]:
        return [self.processed_dir.joinpath("processed_pdb")]

    @property
    def raw_dir(self):
        return self.root.joinpath("raw/")

    @property
    def processed_dir(self):
        return self.root.joinpath("processed/")

    def download(self, force=False) -> None:

        if not force and all([file.exists() for file in self.processed_file_names]):
            print("Downloading skipped - Processed files exist")
            return

        print("Downloading ...")
        if type(self.pdbs) is list:
            fetchPDBviaHTTP(self.pdbs, compressed=True)
        else:
            fetchPDBviaHTTP(self.pdbs.index.values.tolist(), compressed=True)

    def process(self, force=False) -> None:

        if not force and all([file.exists() for file in self.processed_file_names]):
            print("Processing skipped - Processed files exist")
            return
        
        if type(self.pdbs) is list:
            kwargsList = [{"pdb": parsePDB(pdb)} for pdb in self.pdbs]
        else:
            def helper(pdb, metaData):
                kwargs = metaData.to_dict()
                kwargs["pdb"] = parsePDB(pdb)
                return kwargs
            kwargsList = [helper(pdb, metaData) for pdb, metaData in self.pdbs.iterrows()]

        print(kwargsList)
        exit(0)

        if self.pool is None:
            dataList = [self.pre_transform(**kwargs) for kwargs in kwargsList]
        else:
            kwargsList = [kwargs.update({"func": self.pre_transform}) for kwargs in kwargsList]
            dataList = self.pool.map(wrapper, kwargsList)


        if not self.pre_filter is None:
            dataList = list(filter(self.pre_filter, dataList))

        data, slices = self.collate(dataList)
        torch.save((data, slices), self.processed_file_names[0])


class CDRInMemoryDataset(PDBInMemoryDataset):

    def __init__(
        self, 
        summary_file : Path,
        root : Path,
        pre_filter,
        xExtracter = None,
        edgeAttrExtracter = None,
        edgeFilter = None,
        **kwargs
    ) -> None:

        # Extract meta data from description file
        fields = ["pdb", "Hchain", "Lchain", "antigen_chain"]
        summary = pd.read_csv(summary_file, usecols=fields, delimiter="\t")

        concat = lambda x: x.dropna().tolist()
        summary = summary.groupby(summary["pdb"]).agg({"Hchain": concat, "Lchain": concat, "antigen_chain": concat})

        # Set up pre transform
        pre_transform = PDBBuilder(
          seqExtracter=GetSequenceCDR(),
          xExtracter=xExtracter,
          yExtracter=GetSequence(),
          edgeAttrExtracter=edgeAttrExtracter,
          edgeFilter=edgeFilter
        )

        super().__init__(root=root, 
                         pdbs=summary.head(10),
                         pre_filter=pre_filter,
                         pre_transform=pre_transform,
                         **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()
