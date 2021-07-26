# General imports
import pandas as pd
from pathlib import Path
from prody import fetchPDBviaHTTP, pathPDBFolder, parsePDB, AtomGroup
from typing import List, Optional

# PyTorch imports
import torch

# Local imports
from ProteinPairsGenerator.Data import BaseDataset
from ProteinPairsGenerator.PreProcessing import *

class PDBDataset(BaseDataset):

    def __init__(
        self,
        root : Path,
        pbdDict : dict,
        features : List[FeatureModule] = [],
        batchSize : Optional[int] = None,
        pdbFolder : Optional[Union[str, Path]] = None
    ) -> None:

        # Set pdb dict
        self.pbdDict = pbdDict
        
        # Set up PDB folder
        self.pdbFolder = pdbFolder if not pdbFolder is None else root.joinpath("raw")
        self.pdbFolder.mkdir(parents=True, exist_ok=True)
        pathPDBFolder(folder=self.pdbFolder, divided=False)

        # Set up preprocessing
        gen = DataGeneratorList(features=features, batchSize=batchSize)

        # Initialize supra class
        super().__init__(root, [Path(k) for k in self.pbdDict.keys()], gen)

    def download(self, force=False) -> None:

        if not force and self.finished_processing:
            print("Downloading skipped - Processed files exist")
            return

        print("Downloading ...")
        for k in self.processing_queue:
            fetchPDBviaHTTP(*self.pdbDict[k.name], compressed=True)

    def process(self, force=False) -> None:

        if not force and self.finished_processing:
            print("Processing skipped - Processed files exist")
            return

        # Create list of DataPDB from the dataframe containing all pdbs
        while len(self.processing_queue) > 0:

            # Get input file and output directory
            outDir = self.processing_queue.pop()

            # Make output directory
            outDir.mkdir(parents=True, exist_ok=True)

            # Iterate over chunks
            for dataList in self.pre_transform(self.pbdDict[outDir.name]):

                # Coalate and save each chunk
                data, slices = self.collate(dataList)
                newName = self.newProcessedFile()
                torch.save((data, slices), outDir.joinpath(newName))

            # Add to finished subsets
            self.subsets.append(outDir)

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
