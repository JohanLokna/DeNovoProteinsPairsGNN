# General imports
import copy
from itertools import chain, repeat, product
import json
import os
from pathlib import Path
from random import choices
from shutil import copyfile
import string
from typing import List, Union, Tuple

# PyTorch imports
import torch
from torch.utils.data import Subset
from torch_geometric.data import Dataset

# Local imports
from ProteinPairsGenerator.PreProcessing import *

class ProteinNetDataset(Dataset):

    def __init__(
        self,
        root : Path,
        subsets : List[Path],
        features : List[FeatureModule] = [],
        batchSize : Union[int, None] = None,
        caspVersion : int = 12,
    ) -> None:

        # Pre-initialize root to avoid erros
        root.mkdir(parents=True, exist_ok=True)
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
        Dataset.__init__(self, root=root, transform=None, pre_transform=gen, pre_filter=None)

        # Se up indexing
        self.setUpIndexing()

        # Load first file
        self.pageFault(0, 0)

    @property
    def processed_dir(self):
        return self.root.joinpath("processed")

    @property
    def processed_file_names(self) -> List[Path]:
        return list(chain(*[self.getFilesInSubset(subset) for subset in self.subsets])) \
             + [subset.joinpath("dummy") for subset in self.processing_queue]

    @property
    def finished_processing(self) -> bool:
        return len(self.processing_queue) == 0

    @property
    def casp_dir(self):
        return self.raw_dir.joinpath("casp{}".format(self.caspVersion))

    @property
    def raw_dir(self):
        return self.root.joinpath("raw")

    @property
    def raw_file_names(self) -> List[Path]:
        return [self.getCaspFile(f) for f in self.processing_queue]

    @property
    def finished_download(self) -> bool:
        return all([f.exists() for f in self.raw_file_names])

    @property
    def meta_files(self) -> List[str]:
        return ["pre_transform.pt", "pre_filter.pt", "subset.json"]

    def newProcessedFile(self):
        while True:
            newName = ''.join(choices(string.ascii_uppercase + string.digits, k=self.nameSize)) + ".pt"
            if newName not in [f.name for f in self.processed_file_names]:
                return newName

    def getFilesInSubset(self, p : Path) -> List[Path]:
        return [f for f in p.iterdir() if str(f.name) not in self.meta_files]

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

    def setUpIndexing(self):
        self.indexingDict = {}
        self.subsetMapping = {}
        self.filesMapping = {}
        self.totalLength = 0

        for i, subset in enumerate(self.subsets):

            # Update bijective mapping
            self.subsetMapping[i] = subset
            self.subsetMapping[subset] = i

            # Test if meta data is provided
            metaData = subset.joinpath("subset.json")
            if metaData.exists():
                localDict = json.load(metaData.open())

            # Otherwise construct the metadata
            else:
                localDict = {}
                for j, f in enumerate(self.getFilesInSubset(subset)):
                    _, slices = torch.load(f)
                    nElements = list(slices.values())[0].shape[0] - 1
                    localDict[j] = (f.name, nElements)
                json.dump(localDict, metaData.open(mode="w"))
          
            localDict =  {int(k): v for k, v in localDict.items()}
            for j, (name, nElements) in localDict.items():
                f = subset.joinpath(name)
                self.totalLength += nElements
                self.filesMapping[(i, j)] = f
                self.filesMapping[f] = (i, j)
            self.indexingDict[i] = localDict

    def pageFault(self, subsetIdx, fileIdx):
        f = self.filesMapping[(subsetIdx, fileIdx)]
        self.data, self.slices = torch.load(f, map_location="cpu")
        self.subsetIdx, self.fileIdx = subsetIdx, fileIdx

    def len(self):
        return self.totalLength

    def __getitem__(self, idx):

        # Switch between tuple for selection one index
        # and list to get batch
        # print(idx)

        if isinstance(idx, tuple):
            return self.index_select(idx)
        
        elif isinstance(idx, list):
            return [self.index_select(subIdx) for subIdx in idx]
        
        else:
            raise NotImplementedError

    def index_select(self, allIdx):

        subsetIdx, fileIdx, idx = allIdx
        if (subsetIdx, fileIdx) != (self.subsetIdx, self.fileIdx):
            self.pageFault(subsetIdx, fileIdx)
        
        data = self.data.__class__()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                cat_dim = self.data.__cat_dim__(key, item)
                if cat_dim is None:
                    cat_dim = 0
                s[cat_dim] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        return data

    def getSubsetIndecies(self, allSubsets : Union[Path, List[Path]]) -> List[Tuple]:
        indecies = []
        for subset in allSubsets if type(allSubsets) is list else [allSubsets]:
            for f in self.getFilesInSubset(subset):
                i, j = self.filesMapping[f]
                _, n = self.indexingDict[i][j]
                indecies += [(i, j, k) for k in range(n)]
        return indecies

    def getSubset(self, subset : Path) -> Subset:
        return Subset(self, self.getSubsetIndecies(subset))

    @property
    def __indices__(self):
        return list(chain(*[self.getSubsetIndecies(subset) for subset in self.subsets]))

    @__indices__.setter
    def __indices__(self, value):
        pass

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

        # Coords
        coords = StackedFeatures(
            featureName = "coords",
            dependencies = [coordsN, coordsCA, coordsC]
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

        # Construct edge attributes
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
        constraintMinLength = Constraint(
            featureName = "constraintMinLength",
            constraint = lambda attr: attr.shape[0] > 50,
            dependencies = [nodeAttr]
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

        return [nodeAttr, edgeAttr, edgeIdx, title, mask, tape, coords, constraintMaxSize, constraintMinLength]

    @staticmethod
    def collate(data_list):
        """Collates a python list of data objects to the internal storage format"""
        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if isinstance(item[key], torch.Tensor) and item[key].dim() > 0:
                cat_dim = item.__cat_dim__(key, item[key])
                cat_dim = 0 if cat_dim is None else cat_dim
                s = slices[key][-1] + item[key].size(cat_dim)
            else:
                s = slices[key][-1] + 1
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            item = data_list[0][key]
            if isinstance(item, torch.Tensor) and len(data_list) > 1:
                if item.dim() > 0:
                    cat_dim = data.__cat_dim__(key, item)
                    cat_dim = 0 if cat_dim is None else cat_dim
                    data[key] = torch.cat(data[key], dim=cat_dim)
                else:
                    data[key] = torch.stack(data[key])
            elif isinstance(item, torch.Tensor):  # Don't duplicate attributes...
                data[key] = data[key][0]
            elif isinstance(item, int) or isinstance(item, float):
                data[key] = torch.tensor(data[key])

            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices
