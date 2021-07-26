# General imports
import os
from pathlib import Path
from typing import List, Union

# PyTorch imports
import torch

# Local imports
from ProteinPairsGenerator.Data import BaseDataset
from ProteinPairsGenerator.PreProcessing import *

class CaspDataset(BaseDataset):

    def __init__(
        self,
        root : Path,
        subsets : List[Path],
        features : List[FeatureModule] = [],
        batchSize : Union[int, None] = None,
        caspVersion : int = 12,
    ) -> None:

        # Ensure that the used casp version is valid
        self.caspVersion = caspVersion
        if not self.caspVersion in list(range(7, 12 + 1)):
            raise Exception("CASP version is invalid")

        # Set up preprocessing
        gen = DataGeneratorFile(features=features, batchSize=batchSize)

        # Initialize supra class
        super().__init__(root, subsets, gen)

    @property
    def casp_dir(self):
        return self.raw_dir.joinpath("casp{}".format(self.caspVersion))

    @property
    def raw_file_names(self) -> List[Path]:
        return [self.getCaspFile(f) for f in self.processing_queue]

    def getCaspFile(self, p : Path) -> Path:
        return self.casp_dir.joinpath(p.name)

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
        while len(self.processing_queue) > 0:

            # Get input file and output directory
            outDir = self.processing_queue.pop()
            inPath = self.getCaspFile(outDir)

            # Make output directory
            outDir.mkdir(parents=True, exist_ok=True)

            # Iterate over chunks
            for dataList in self.pre_transform(inPath):

                # Coalate and save each chunk
                data, slices = self.collate(dataList)
                newName = self.newProcessedFile()
                torch.save((data, slices), outDir.joinpath(newName))

            # Add to finished subsets
            self.subsets.append(outDir)

    @staticmethod
    def getGenericFeatures(device : str = "cuda:0" if torch.cuda.is_available() else "cpu"):

        reader = ProteinNetRecord()

        # Get sequence attributes
        nodeAttr = reader.getPrimary("seq")

        # Construct coordinates & distances
        coordsCA = reader.getCoordsCA("coordsCA")
        coordsN = reader.getCoordsN("coordsN")
        coordsC = reader.getCoordsC("coordsC")
        cartDist = CartesianDistances(dependencies=[coordsCA])
        seqDist = SequenceDistances(dependencies=[nodeAttr])

        # Get coords and scale them to fit Ingraham Model
        coords = StackedFeatures(
            featureName = "coords",
            dependencies = [coordsN, coordsCA, coordsC],
            dim = 1
        )

        coordsScaled = Normalize(
            bias = 0,
            scale = 100,
            featureName = "coordsScaled",
            dependencies = [coords]
        )

        # Construct edge relations
        closeNeighbours = CloseNeighbours(
            threshold = 1200,
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
            bias = torch.Tensor([7.5759e+02, 1.4498e-06]), # Should be zero in second coordinate but roundoff error
            scale = torch.Tensor([368.0696, 116.6342]),
            featureName = "edge_attr",
            dependencies = [edgeAttrUnnormal]
        )

        # Construct title
        title = reader.getId("title")

        # Restrict sequence lenght
        constraintMaxSize = Constraint(
            featureName = "constraintMaxSize",
            constraint = lambda attr: attr.shape[0] < 200000,
            dependencies = [edgeAttr]
        )

         # Get only get sequences where all positions are well defined
        validityMask = reader.getMask("mask")
        constraintAllValid = Constraint(
            featureName = "constraintAllValid",
            constraint = lambda x: torch.all(x).item(),
            dependencies = [validityMask]
        )

        # Get BERT masking
        mask = MaskBERT(
            dependencies = [nodeAttr],
            nMasks = 4
        )

        # Get TAPE annotation
        tape = TAPEFeatures(
            dependencies = [mask],
            device = device
        )

        return [nodeAttr, edgeAttr, edgeIdx, title, mask, tape, coordsScaled, constraintMaxSize, constraintAllValid]
