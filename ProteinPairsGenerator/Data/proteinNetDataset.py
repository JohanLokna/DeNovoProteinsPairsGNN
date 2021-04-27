# General imports
from itertools import chain
import os
from pathlib import Path
from random import choices
import string
from typing import List, Union

# PyTorch imports
import torch
from torch_geometric.data import InMemoryDataset

# Local imports
from ProteinPairsGenerator.PreProcessing import *

class ProteinNetDataset(InMemoryDataset):

    def __init__(
        self,
        root : Path,
        subsets : List[Path],
        features : List[FeatureModule],
        batchSize : Union[int, None] = None,
        caspVersion : int = 12,
    ) -> None:

        # Pre-initialize root to avoid erros
        self.root = root

        # Lenght og name used for storing batched files
        self.nameSize = 6

        # Ensure that the used casp version is valid
        self.caspVersion = caspVersion
        if not self.caspVersion in list(range(7, 12 + 1)):
            raise Exception("CASP version is invalid")

        # Differentiate between new directories to be created
        # and existing directories
        self.processing_queue = []
        self.subsets = []
        for p in subsets:
            if not type(p) is Path:
                raise Exception("All subsets must be of type Path")
            if p.exists():
                if p.is_dir():
                    self.subsets.append(p)
                else:
                    raise Exception("All exisiting subsets must refer to a directory")
            else:
                self.processing_queue.append(self.processed_dir.joinpath(p.name))
        
        # Set up preprocessing
        gen = DataGeneratorFile(features=features, batchSize=batchSize)

        # Initialize super class and complete set up
        super().__init__(root=root, transform=None, pre_transform=gen, pre_filter=None)
        self.data, self.slices = None, None

    @property
    def processed_dir(self):
        return self.root.joinpath("processed")

    @property
    def processed_file_names(self) -> List[Path]:
        return list(chain(*[getFilesInSubset(subset) for subset in self.subsets])) \
             + list(chain(*[subset.joinpath("dummy") for subset in self.processing_queue]))

    @property
    def raw_dir(self):
        return self.root.joinpath("raw")

    @property
    def casp_dir(self):
        return self.raw_dir.joinpath("casp{}".format(self.caspVersion))

    @property
    def raw_file_names(self) -> List[Path]:
        return [self.casp_dir.joinpath(f.name) for f in self.processing_queue]

    @property
    def finished_download(self) -> bool:
        return all([f.exists() for f in self.raw_file_names])

    def newProcessedFile(self):
        while True:
            newName = ''.join(choices(string.ascii_uppercase + string.digits, k=self.nameSize)) + ".pt"
            if newName not in [f.name for f in self.processed_file_names]:
                return newName

    def getFilesInSubset(self, p : Path) -> List[Path]:
        return [f for f in p.iterdir() if str(f.name) not in ["pre_transform.pt", "pre_filter.pt"]]

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
        for inPath, outDir in zip(self.raw_file_names, self.processing_queue):
            for dataList in self.pre_transform(inPath):
                
                # Coalate and save
                data, slices = self.collate(dataList)
                newName = self.newProcessedFile()
                torch.save((data, slices), outDir.joinpath(newName))

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
