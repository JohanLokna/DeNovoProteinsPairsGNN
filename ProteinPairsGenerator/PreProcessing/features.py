# General imports
from pathlib import Path
from prody import AtomGroup, ANM
from prody.atomic.select import Select
from typing import List, IO
import warnings

# PyTorch imports
import torch

#Local imports
from ProteinPairsGenerator.utils.amino_acids import seq_to_tensor, \
                                                    AMINO_ACIDS_MAP, AMINO_ACIDS_BASE, \
                                                    CDRS_HEAVY, CDRS_LIGHT, \
                                                    CHAINS_MAP
from ProteinPairsGenerator.utils.cdr import getHeavyCDR, getLightCDR
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

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def preFilter(self, *args, **kwargs) -> bool:
        raise NotImplementedError

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

class SequencePDB(FeatureModule):

    def __init__(
        self,
        featureName : str = "seq"
    ) -> None:
        super().__init__(featureName)
    
    def forward(
        self, 
        pdb : AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:
        
        # Get sequence
        return seq_to_tensor(pdb.getSequence(), mapping=AMINO_ACIDS_MAP)

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class SequencePDBwithCDR(FeatureModule):

    def __init__(
        self,
        featureName : str = "seqCDR",
        hmmerpath : str = "/usr/bin/",
        dependencies : List[FeatureModule] = []
    ) -> None:
        
        self.hmmerpath = hmmerpath

        # Set up dependecies
        if len(dependencies) == 0:
            dependencies = [Sequence()]
        elif len(dependencies) != 1 or dependencies[0].featureName != "seq":
            warnings.warn("Dependencies in SequenceCDR might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        pdb : AtomGroup,
        Lchain : List[str] = [],
        Hchain : List[str] = [],
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Get sequence
        seq = self.dependencies[0].data

        # Mask CDR in light chains in seq
        for c in Lchain:
          idx = Select().getIndices(pdb, "chain {}".format(c))
          for i, cdr in enumerate(getLightCDR(pdb.select("chain {}".format(c)).getSequence(), hmmerpath=self.hmmerpath)):
            seq[idx[cdr]] = AMINO_ACIDS_MAP[CDRS_LIGHT[i]]

        # Mask CDR in heavy chains in seq
        for c in Hchain:
          idx = Select().getIndices(pdb, "chain {}".format(c))
          for i, cdr in enumerate(getHeavyCDR(pdb.select("chain {}".format(c)).getSequence(), hmmerpath=self.hmmerpath)):
            seq[idx[cdr]] = AMINO_ACIDS_MAP[CDRS_HEAVY[i]]

        return seq

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs \
           and "Lchain" in kwargs \
           and "Hchain" in kwargs \
           and set(kwargs["Lchain"] + kwargs["Hchain"]) <= set(kwargs["pdb"].getChids()) \
           and set(kwargs["pdb"].getSequence()) <= set(AMINO_ACIDS_BASE)


class Modes(FeatureModule):

    def __init__(
        self,
        featureName : str = "modes",
        nModes : int = 20,
        maxNodes : int = 5000
    ) -> None:
        self.nModes = nModes
        self.maxNodes = maxNodes
        super().__init__(featureName)

    def forward(
        self, 
        pdb : AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # ANM set up mode calculations
        pdbANM = ANM(pdb)
        pdbANM.buildHessian(pdb)
        pdbANM.calcModes(self.nModes)

        # Make into array and reshape to [numAtoms, -1]
        modes = torch.from_numpy(pdbANM.getArray())
        return modes.view(pdb.numAtoms(), -1)

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs and kwargs["pdb"].numAtoms() <= self.maxNodes


class CartesianCoordinatesPDB(FeatureModule):

    def __init__(
        self,
        featureName : str = "cartCoords"
    ) -> None:
        super().__init__(featureName)
    
    def forward(
        self, 
        pdb : AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Get Cartesian coordinates
        return torch.from_numpy(pdb.getCoordsets(0))

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class CartesianDistances(FeatureModule):

    def __init__(
        self,
        featureName : str = "cartDist",
        dependencies : List[FeatureModule] = []
    ) -> None:
        
        # Set up dependecies
        if len(dependencies) == 0:
            dependencies = [CartesianCoordinates()]
        elif len(dependencies) != 1:
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
            warnings.warn("Dependencies in SequenceCDR might be errornous!", UserWarning)

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
        dependencies : List[FeatureModule] = []
    ) -> None:

        if len(dependencies) == 0:
            warnings.warn("Dependencies in StackedFeatures might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return torch.stack([d.data for d in self.dependencies], dim=-1)

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
            "getCoordsC": lambda x: ProteinNetField(featureName=x, fieldName="C", dependencies=[self])
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
        dependecies : List[FeatureModule] = []
    ) -> None:

        super().__init__(featureName, save = False)
        self.constraint = constraint

    def forward(
        *args,
        **kwargs
    ) -> torch.Tensor:
        pass
        
    def preFilter(self, *args, **kwargs) -> bool:
        return self.runDependencies(*args, **kwargs) \
           and self.constraint(*[d.data for d in self.dependencies])
