# General imports
import pandas as pd
from pathlib import Path
from prody import fetchPDBviaHTTP, pathPDBFolder, parsePDB, AtomGroup
from typing import List, Mapping, Callable, Union, Generator, Any

# PyTorch imports
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

# Local imports
from ProteinPairsGenerator.PreProcessing import *

class PDBDataset(InMemoryDataset):

    def __init__(
        self,
        root : Union[Path, None],
        pdbs : pd.DataFrame,
        features : List[FeatureModule],
        device : str = "cuda:0" if torch.cuda.is_available() else "cpu",
        pdbFolder : Union[Path, None] = None
    ) -> None:

        # Set up root
        root.mkdir(parents=True, exist_ok=True)

        # Set up PDB
        self.pdbs = pdbs

        # Set up PDB folder
        self.pdbFolder = pdbFolder if not pdbFolder is None else root.joinpath("raw")
        self.pdbFolder.mkdir(parents=True, exist_ok=True)
        pathPDBFolder(folder=self.pdbFolder, divided=False)
        
        # Set up preprocessing
        gen = DataGeneratorList(features = features)

        # Initialize super class and complete set up
        super().__init__(root=root, transform=None, pre_transform=gen, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def processed_dir(self):
        return self.root.joinpath("processed")

    @property
    def processed_file_names(self) -> List[Path]:
        return [self.processed_dir.joinpath("processed.pt") ]    

    @property
    def raw_dir(self):
        return self.root.joinpath("raw")

    @property
    def raw_file_names(self) -> List[Path]:
            return [self.raw_dir.joinpath(pdb + ".pdb.gz") for pdb in self.pdbs.index.values.tolist()]

    @property
    def finished_processing(self) -> bool:
          return all([f.exists() for f in self.processed_file_names])


    def download(self, force=False) -> None:

        if not force and self.finished_processing:
            print("Downloading skipped - Processed files exist")
            return

        print("Downloading ...")
        fetchPDBviaHTTP(*self.pdbs.index.values.tolist(), compressed=True)

    def process(self, force=False) -> None:

        if not force and self.finished_processing:
            print("Processing skipped - Processed files exist")
            return

        # Create list of DataPDB from the dataframe containing all pdbs
        dataList = self.pre_transform(
          [{"pdb": parsePDB(pdb).ca, **metaData.to_dict()} for pdb, metaData in self.pdbs.iterrows()]
        )

        # Coalate and save
        data, slices = self.collate(dataList)
        torch.save((data, slices), self.processed_file_names[0])

    @staticmethod
    def getGenericFeatures():

        # Get sequence attributes
        nodeAttr = SequencePDB(featureName = "x")

        # Construct coordinates & distances
        coords = CartesianCoordinatesPDB()
        cartDist = CartesianDistances(dependencies=[coords])
        seqDist = SequenceDistances(dependencies=[nodeAttr])

        # Construct edge relations
        closeNeighbours = CloseNeighbours(
            threshold = 12,
            dependencies = [cartDist]
        )
        edgeIdx = EdgeIndecies(
            featureName = "edge_index",
            dependencies = [closeNeighbours]
        )

        # Construct edge attributes
        stackedDist = StackedFeatures(
            dependencies = [cartDist, seqDist]
        )
        edgeAttr = EdgeAttributes(
            featureName = "edge_attr",
            dependencies = [stackedDist, closeNeighbours]
        )

        # Construct title
        title = Title()

        return [nodeAttr, edgeAttr, edgeIdx, title]
