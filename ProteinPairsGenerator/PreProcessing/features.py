# General imports
from prody import AtomGroup, parsePDB
from typing import List, IO
import warnings

# PyTorch imports
import torch

#Local imports
from ProteinPairsGenerator.utils.amino_acids import seq_to_tensor, AMINO_ACIDS_MAP, AMINO_ACIDS_BASE
from ProteinPairsGenerator.BERTModel import maskBERT, TAPEAnnotator
from ProteinPairsGenerator.PreProcessing.proteinNetParser import readPotein

class FeatureModule:

    def __init__(
        self, 
        featureName : str,
        dependencies : List = [],
        save : bool = True
    ) -> None:

        # Set up structure
        self.featureName = featureName
        self.dependencies = dependencies
        self.save = save
        self.clear()

    @property
    def data(self):
        return self.data_

    @data.setter
    def data(self, value):
        self.data_ = value

    @property
    def dataId(self):
        return {"args": self.args_, "kwargs": self.kwargs_}

    def setDataId(self, *args, **kwargs):
        self.args_ = args
        self.kwargs_ = kwargs

    def testDataId(self, *args, **kwargs):
        return self.args_ == args and self.kwargs_ == kwargs

    def clear(self):
        self.data = None
        self.setDataId()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies] + [True])

    def runDependencies(self, *args, **kwargs) -> bool:

        for feature in self.dependencies:
            if not feature(*args, **kwargs):
                return False
        return True

    def __call__(self, *args, **kwargs) -> bool:

        if self.testDataId(*args, **kwargs):
            return True

        if not self.preFilter(*args, **kwargs):
            return False

        if not self.runDependencies(*args, **kwargs):
            return False

        self.data = self.forward(*args, **kwargs)
        self.setDataId(*args, **kwargs)
        return True


# Modules used for computing features

class ProdyPDB(FeatureModule):

    def __init__(
        self,
        featureName : str = "prodyPDB"
    ) -> None:
        super().__init__(featureName)
    
    def forward(
        self, 
        pdb,
        *args,
        **kwargs
    ) -> AtomGroup:
        
        # Get sequence
        return parsePDB(pdb).select("chain A")


class ProdySelect(FeatureModule):

    def __init__(
        self,
        selectSequence : str,
        featureName : str = "prodySelect",
        dependencies : List[FeatureModule] = []
    ) -> None:

        self.selectSequence = selectSequence

        # Set up dependecies
        if len(dependencies) != 1:
            warnings.warn("Dependencies in ProdySelect might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> AtomGroup:
        return self.dependencies[0].data.select(self.selectSequence)


class ProdyBackboneCoords(FeatureModule):

    def __init__(
        self,
        featureName : str = "prodyBackboneCoords",
        dependencies : List[FeatureModule] = []
    ) -> None:

        # Set up dependecies
        if len(dependencies) != 1:
            warnings.warn("Dependencies in ProdyBackboneCoords might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        
        # Get backbone coords
        coords = self.dependencies[0].data.backbone.getCoordsets(0)

        # Reshape to [seqDim, atomDim, spatialDim]
        # Atm ordering in atomDim is: N, CA, C, O
        return torch.from_numpy(coords).reshape((-1, 4, 3)).float()


class ProdySequence(FeatureModule):

    def __init__(
        self,
        featureName : str = "seq",
        dependencies : List[FeatureModule] = []
    ) -> None:

        # Set up dependecies
        if len(dependencies) != 1:
            warnings.warn("Dependencies in ProdySequence might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self, 
        pdb : AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:
        
        # Get sequence
        return seq_to_tensor(self.dependencies[0].data.getSequence(), mapping=AMINO_ACIDS_MAP)


class ProdyCartesianCoordinates(FeatureModule): 

    def __init__(
        self,
        featureName : str = "cartCoords",
        dependencies : List[FeatureModule] = []
    ) -> None:

        # Set up dependecies
        if len(dependencies) != 1:
            warnings.warn("Dependencies in ProdyCartesianCoordinates might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Get Cartesian coordinates
        return torch.from_numpy(self.dependencies[0].data.getCoordsets(0)).float()


class ProdyTitle(FeatureModule): 

    def __init__(
        self,
        featureName : str = "title",
        dependencies : List[FeatureModule] = []
    ) -> None:

        # Set up dependecies
        if len(dependencies) != 1:
            warnings.warn("Dependencies in ProdyTitle might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> str:

        # Get Cartesian coordinates
        return self.dependencies[0].data.getTitle()


class CartesianDistances(FeatureModule):

    def __init__(
        self,
        featureName : str = "cartDist",
        dependencies : List[FeatureModule] = []
    ) -> None:
        
        # Set up dependecies
        if len(dependencies) != 1:
            warnings.warn("Dependencies in CartesianDistances might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Compute caresian distances
        coords = self.dependencies[0].data
        return torch.cdist(coords, coords).squeeze(0)


    def preFilter(self, *args, **kwargs) -> bool:
        return [d.preFilter(*args, **kwargs) for d  in self.dependencies]


class SequenceDistances(FeatureModule):

    def __init__(
        self,
        featureName : str = "seqtDist",
        dependencies : List[FeatureModule] = []
    ) -> None:

        # Set up dependecies
        if len(dependencies) == 0:
            dependencies = [Sequence()]
        elif len(dependencies) != 1:
            warnings.warn("Dependencies in SequenceDistances might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Compute sequence distances
        size = len(self.dependencies[0].data)
        x = torch.arange(size).view(-1, 1).expand(size, 2)
        return x[:, 0] - x[:, 1].view(-1, 1)

    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])


class CloseNeighbours(FeatureModule):

    def __init__(
        self,
        threshold,
        featureName : str = "closeNeighbours",
        dependencies : List[FeatureModule] = []
    ) -> None:
        self.threshold = threshold
        
        # Set up dependecies
        if len(dependencies) == 0:
            dependencies = [CartesianDistances()]
        elif len(dependencies) != 1:
            warnings.warn("Dependencies in CloseNeighbours might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return self.dependencies[0].data < self.threshold

    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])


class EdgeIndecies(FeatureModule):

    def __init__(
        self,
        featureName : str = "edgeIndecies",
        dependencies : List[FeatureModule] = []
    ) -> None:

        if len(dependencies) != 1:
            warnings.warn("Dependencies in EdgeIndecies might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return torch.stack(torch.where(self.dependencies[0].data), dim=0)

    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])


class EdgeAttributes(FeatureModule):

    def __init__(
        self,
        featureName : str = "edgeAttributes",
        dependencies : List[FeatureModule] = []
    ) -> None:

        if len(dependencies) != 2:
            warnings.warn("Dependencies in EdgeAttributes might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    @property
    def attributes(self):
        return self.dependencies[0].data.squeeze(dim=1)
    
    @property
    def mask(self):
        return self.dependencies[1].data.squeeze(dim=1)

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return self.attributes[self.mask]

    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])


class StackedFeatures(FeatureModule):

    def __init__(
        self,
        featureName : str = "stack",
        dependencies : List[FeatureModule] = [],
        dim : int = -1
    ) -> None:

        if len(dependencies) == 0:
            warnings.warn("Dependencies in StackedFeatures might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
        self.dim = dim

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return torch.stack([d.data for d in self.dependencies], dim=self.dim)

    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])


class Title(FeatureModule):

    def __init__(
        self,
        featureName : str = "title"
    ) -> None:

        super().__init__(featureName)

    def forward(
        self,
        pdb : AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return pdb.getTitle()

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class ProteinNetField(FeatureModule):

    def __init__(
        self,
        featureName : str,
        fieldName : str,
        dependencies : List[FeatureModule] = []
    ) -> None:

        self.fieldName = fieldName
   
        # Set up dependecies
        if len(dependencies) == 0:
            dependencies = [ProteinNetRecord()]
        elif len(dependencies) != 1:
            warnings.warn("Dependencies in ProteinNetField might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)

    def forward(
        self,
        inFile : IO[str],
        *args,
        **kwargs
    ) -> torch.Tensor:
        return self.dependencies[0].data[self.fieldName]

    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])

from copy import copy

class ProteinNetRecord(FeatureModule):

    def __init__(
        self,
        featureName : str = "proteinnetRecord"
    ) -> None:

        # Update getters for fields
        self.__dict__.update({
            "getId": lambda x: ProteinNetField(featureName=x, fieldName="id", dependencies=[self]),
            "getPrimary": lambda x: ProteinNetField(featureName=x, fieldName="primary", dependencies=[self]),
            "getCoordsN": lambda x: ProteinNetField(featureName=x, fieldName="N", dependencies=[self]),
            "getCoordsCA": lambda x: ProteinNetField(featureName=x, fieldName="CA", dependencies=[self]),
            "getCoordsC": lambda x: ProteinNetField(featureName=x, fieldName="C", dependencies=[self]),
            "getMask": lambda x: ProteinNetField(featureName=x, fieldName="mask", dependencies=[self])
        })

        super().__init__(featureName)

    def forward(
        self,
        inFile : IO[str],
        *args,
        **kwargs
    ) -> torch.Tensor:

        record = readPotein(inFile)

        if record is None:
            raise EOFError
        else:
            return record

    def preFilter(self, *args, **kwargs) -> bool:
        return "inFile" in kwargs

class Constraint(FeatureModule):

    def __init__(
        self,
        constraint,
        featureName : str = "constraint",
        dependencies : List[FeatureModule] = []
    ) -> None:

        super().__init__(featureName, dependencies=dependencies, save = False)
        self.constraint = constraint

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        pass
        
    def preFilter(self, *args, **kwargs) -> bool:
        return self.runDependencies(*args, **kwargs) \
           and self.constraint(*[d.data for d in self.dependencies])

class Normalize(FeatureModule):

    def __init__(
        self,
        bias,
        scale,
        featureName : str = "normalize",
        dependencies : List[FeatureModule] = []
    ) -> None:

        if len(dependencies)!= 1:
            warnings.warn("Dependencies in Normalize might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
        self.bias = bias
        self.scale = scale

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return (self.dependencies[0].data - self.bias) / self.scale
        
    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])


class MaskBERT(FeatureModule):

    def __init__(
        self,
        featureName : str = "maskBERT",
        dependencies : List[FeatureModule] = [],
        nMasks : int = 1,
        subMatrix : torch.Tensor = torch.ones(len(AMINO_ACIDS_BASE), len(AMINO_ACIDS_BASE))
    ) -> None:

        if len(dependencies)!= 1:
            warnings.warn("Dependencies in BERTMask might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
        self.nMasks = nMasks
        self.subMatrix = subMatrix

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return [maskBERT(self.dependencies[0].data, self.subMatrix) for _ in range(self.nMasks)]
        
    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])


class TAPEFeatures(FeatureModule):

    def __init__(
        self,
        featureName : str = "TAPE",
        dependencies : List[FeatureModule] = [],
        *args,
        **kwargs
    ) -> None:

        if len(dependencies)!= 1:
            warnings.warn("Dependencies in TAPEFeatures might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
        self.annotator = TAPEAnnotator(*args, **kwargs)

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return [self.annotator([seq])[0] for seq, _ in self.dependencies[0].data]
        
    def preFilter(self, *args, **kwargs) -> bool:
        return all([d.preFilter(*args, **kwargs) for d  in self.dependencies])
