# General imports
import os
import pandas as pd
from pathlib import Path
from prody import fetchPDBviaHTTP, pathPDBFolder, parsePDB, AtomGroup
from typing import List, Mapping, Callable, Union, Generator, Any

# PyTorch imports
import torch
from torch_geometric.data import InMemoryDataset

# Local imports
from ProteinPairsGenerator.PreProcessing import *

class ProteinNetDataset(InMemoryDataset):

    def __init__(
        self,
        root : Union[Path, None],
        inFile : Path,
        features : List[FeatureModule],
        caspVersion : int = 12,
        device : str = "cuda:0" if torch.cuda.is_available() else "cpu"
    ) -> None:

        # Save features
        self.caspVersion = caspVersion
        if not self.caspVersion in list(range(7, 12 + 1)):
            raise Exception("CASP version is invalid")

        self.raw_file = inFile if inFile.exists() else \
                        root.joinpath("raw").joinpath("casp" + str(self.caspVersion)).joinpath(inFile)

        # Set up root
        root.mkdir(parents=True, exist_ok=True)
        
        # Set up preprocessing
        gen = DataGeneratorFile(features = features)

        # Initialize super class and complete set up
        super().__init__(root=root, transform=None, pre_transform=gen, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def processed_dir(self):
        return self.root.joinpath("processed")

    @property
    def processed_file_names(self) -> List[Path]:
        return [self.processed_dir.joinpath("processed.pt")]

    @property
    def finished_processing(self) -> bool:
          return all([f.exists() for f in self.processed_file_names])

    @property
    def raw_dir(self):
        return self.root.joinpath("raw")

    @property
    def raw_file_names(self) -> List[Path]:
            return [self.raw_file]

    @property
    def finished_download(self) -> bool:
          return all([f.exists() for f in self.raw_file_names])

    def download(self, force=False) -> None:

        if not force and (self.finished_download or self.finished_processing):
            print("Downloading skipped - Files exist")
            return

        print("Downloading ...")
        cmd = "cd {};".format(str(self.raw_dir)) \
            + "wget https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp" \
            + "{}.tar.gz;tar -xvf casp{}.tar.gz".format(str(self.caspVersion), str(self.caspVersion))
        exitCode = os.system(cmd)

        if exitCode != 0:
            raise Exception("Error upon download of CASP{}".format(self.caspVersion))

    def process(self, force=False) -> None:

        if not force and self.finished_processing:
            print("Processing skipped - Processed files exist")
            return

        # Create list of DataPDB from the dataframe containing all pdbs
        dataList = self.pre_transform(self.raw_file_names[0])

        # Coalate and save
        data, slices = self.collate(dataList)
        torch.save((data, slices), self.processed_file_names[0])

    @staticmethod
    def getGenericFeatures():

        reader = ProteinNetRecord()

        # Get sequence attributes
        nodeAttr = reader.primary(featureName = "x")

        # Construct coordinates & distances
        coords = reader.CA(featureName = "coordsCA")
        cartDist = CartesianDistances(dependencies=[coords])
        seqDist = SequenceDistances(dependencies=[nodeAttr])

        # Construct edge relations
        closeNeighbours = CloseNeighbours(
            threshold = 1200,
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
        title = reader.id(featureName = "title")

        return [nodeAttr, edgeAttr, edgeIdx, title]
