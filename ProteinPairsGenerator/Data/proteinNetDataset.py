# General imports
import os
import pandas as pd
from pathlib import Path
from prody import fetchPDBviaHTTP, pathPDBFolder, parsePDB, AtomGroup
from random import random
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
        device : str = "cuda:0" if torch.cuda.is_available() else "cpu",
        nameSize : int = 6
    ) -> None:

        # Save features
        self.nameSize = nameSize

        self.caspVersion = caspVersion
        if not self.caspVersion in list(range(7, 12 + 1)):
            raise Exception("CASP version is invalid")

        self.raw_file = inFile if inFile.exists() else \
                        root.joinpath("raw").joinpath("casp" + str(self.caspVersion)).joinpath(inFile)

        # Set up root
        root.mkdir(parents=True, exist_ok=True)
        if root.joinpath("processed").exists():
              self.processed_names_ = [f.name for f in root.joinpath("processed").iterdir() \
                                       if str(f.name) not in ["pre_transform.pt", "pre_filter.pt"]]
        else:
              self.processed_names_ = []
        
        # To ensure processing
        if len(self.processed_names_) == 0:
            self.newProcessedFile()
        
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
        return self.processed_names_

    @property
    def finished_processing(self) -> bool:
          return False

    @property
    def raw_dir(self):
        return self.root.joinpath("raw")

    @property
    def raw_file_names(self) -> List[Path]:
            return [self.raw_file]

    @property
    def finished_download(self) -> bool:
          return all([f.exists() for f in self.raw_file_names])

    def newProcessedFile(self):
        while True:
            newName = ''.join(choices(string.ascii_uppercase + string.digits, k=self.nameSize)) + ".pt"
            newName = self.raw_dir.joinpath(newName)
            if newName not in self.processed_file_names:
                self.processed_file_names.append(newName)
                return


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

        print("Processing")

        # Create list of DataPDB from the dataframe containing all pdbs
        for dataList in self.pre_transform(self.raw_file_names[0]):
            
            # Coalate and save
            data, slices = self.collate(dataList)

            if self.processed_file_names[-1].exists():
                self.newProcessedFile()

            torch.save((data, slices), self.processed_file_names[-1])

    @staticmethod
    def getGenericFeatures():

        reader = ProteinNetRecord()

        # Get sequence attributes
        nodeAttr = reader.getPrimary("x")

        # Construct coordinates & distances
        coords = reader.getCoordsCA("coordsCA")
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
        title = reader.getId("title")

        return [nodeAttr, edgeAttr, edgeIdx, title, coords]
