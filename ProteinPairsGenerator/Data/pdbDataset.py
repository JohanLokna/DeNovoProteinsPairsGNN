# General imports
import pandas as pd
from pathlib import Path
from prody import fetchPDBviaHTTP, pathPDBFolder, confProDy
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
        pdbDict : dict,
        features : List[FeatureModule] = [],
        batchSize : Optional[int] = None,
        pdbFolder : Optional[Union[str, Path]] = None
    ) -> None:

        # Set pdb dict
        self.pdbDict = pdbDict
        
        # Set up PDB folder
        self.pdbFolder = pdbFolder if not pdbFolder is None else root.joinpath("raw")
        self.pdbFolder.mkdir(parents=True, exist_ok=True)
        pathPDBFolder(folder=self.pdbFolder, divided=False)

        # Silence prody
        confProDy(verbosity='none')

        # Set up preprocessing
        gen = DataGeneratorList(features=features, batchSize=batchSize)

        # Initialize supra class
        super().__init__(root, [Path(k) for k in self.pdbDict.keys()], gen)

    @property
    def raw_file_names(self) -> List[Path]:
        return [self.pdbFolder.joinpath(pdb + ".pdb.gz") \
                for _, v in self.pdbDict.items() for pdb in v]

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
            for dataList in self.pre_transform([{"pdb": pdb} for pdb in self.pdbDict[outDir.name]]):

                # Coalate and save each chunk
                data, slices = self.collate(dataList)
                newName = self.newProcessedFile()
                torch.save((data, slices), outDir.joinpath(newName))

            # Add to finished subsets
            self.subsets.append(outDir)

    @staticmethod
    def getGenericFeatures():

        pdb = ProdyPDB()

        # Backbone coords for Ingraham
        backboneCoords = ProdyBackboneCoords("coordsScaled", dependencies=[pdb])

        # Get sequence attributes for Strokach
        nodeAttr = ProdySequence("seq", dependencies=[pdb])

        # Get distances within protein, used for Strokach
        coordsCA = ProdySelect("ca")
        cartDist = CartesianDistances(dependencies=[coordsCA])
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

        # Construct edge attributes and normalize them
        stackedDist = StackedFeatures(
            dependencies = [cartDist, seqDist]
        )
        edgeAttrUnnormal = EdgeAttributes(
            dependencies = [stackedDist, closeNeighbours]
        )
        edgeAttr = Normalize(
            bias = torch.Tensor([7.5759e+00, 1.4498e-08]), # Should be zero in second coordinate but roundoff error
            scale = torch.Tensor([3.680696, 1.166342]),
            featureName = "edge_attr",
            dependencies = [edgeAttrUnnormal]
        )



        return [backboneCoords]
