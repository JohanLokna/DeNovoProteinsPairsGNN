# General imports
from pathlib import Path
from prody import AtomGroup, ANM
from prody.atomic.select import Select
from typing import List
import warnings

# PyTorch imports
import torch

#Local imports
from ProteinPairsGenerator.utils.amino_acids import seq_to_tensor, \
                                                    AMINO_ACIDS_MAP, AMINO_ACIDS_BASE, \
                                                    CDRS_HEAVY, CDRS_LIGHT, \
                                                    CHAIN_NULL, CHAIN_HEAVY, CHAIN_LIGHT, CHAIN_ANTIGEN, \
                                                    CHAINS_MAP
from ProteinPairsGenerator.utils.cdr import getHeavyCDR, getLightCDR


class FeatureModule:

    def __init__(
        self, 
        featureName : str,
        identifier = lambda *args, **kwargs : kwargs["pdb"],
        dependencies : List = [],
    ) -> None:

        # Set up structure
        self.featureName = featureName
        self.identifier = identifier
        self.dependencies = dependencies
        self.clear()

    @property
    def data(self):
        return self.data_

    @data.setter
    def data(self, value):
        self.data_ = value

    @property
    def dataId(self):
        return self.dataId_

    @dataId.setter
    def dataId(self, value):
        self.dataId_ = value

    def clear(self):
        self.data = None
        self.dataId = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def preFilter(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    def runDependencies(self, currId, *args, **kwargs) -> bool:

        for feature in self.dependencies:
            if not feature(*args, **kwargs):
                return False
        return True

    def __call__(self, *args, **kwargs) -> bool:

        currId = self.identifier(*args, **kwargs)
        if self.dataId == currId:
            return True

        if not self.preFilter(*args, **kwargs):
            return False

        if not self.runDependencies(currId, *args, **kwargs):
            return False

        self.data = self.forward(*args, **kwargs)
        self.dataId = currId
        return True


# Modules used for computing features

class Sequence(FeatureModule):

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
        return seq_to_tensor(pdb.getSequence())

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class SequenceCDR(FeatureModule):

    def __init__(
        self,
        featureName : str = "seqCDR",
        hmmerpath : str = "/usr/bin/",
        dependencies : List[FeatureModule] = []
    ) -> None:
        
        self.hmmerpath = hmmerpath

        self.computeSeq = len(dependencies) != 0
        if self.computeSeq and (len(dependencies) != 1 or dependencies[0].featureName != "seq"):
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
        if self.computeSeq:
            seq = seq_to_tensor(pdb.getSequence())
        else:
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


class ChainAnnotation(FeatureModule):

    def __init__(
        self,
        featureName : str = "seqChains"
    ) -> None:

        super().__init__(featureName)
    
    def forward(
        self,
        pdb : AtomGroup,
        Lchain : List[str] = [],
        Hchain : List[str] = [],
        antigen_chain: List[str] = [],
        *args,
        **kwargs
    ) -> torch.Tensor:
        
        # Get sequence
        seq = torch.empty(pdb.numAtoms(), dtype=torch.long).fill_(CHAINS_MAP[CHAIN_NULL])

        # Mask light chains in seq
        for c in Lchain:
          idx = Select().getIndices(pdb, "chain {}".format(c))
          seq[idx] = CHAINS_MAP[CHAIN_LIGHT]

        # Mask heavy chains in seq
        for c in Hchain:
          idx = Select().getIndices(pdb, "chain {}".format(c))
          seq[idx] = CHAINS_MAP[CHAIN_HEAVY]

        # Mask antigen chains in seq
        for c in antigen_chain:
          idx = Select().getIndices(pdb, "chain {}".format(c))
          seq[idx] = CHAINS_MAP[CHAIN_ANTIGEN]

        return seq

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs \
           and "Lchain" in kwargs \
           and "Hchain" in kwargs \
           and set(kwargs["Lchain"] + kwargs["Hchain"]) <= set(kwargs["pdb"].getChids())


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


class CartesianCoordinates(FeatureModule):

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
        print(torch.from_numpy(pdb.getCoordsets(0)).shape)
        return torch.from_numpy(pdb.getCoordsets(0))

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class CartesianDistances(FeatureModule):

    def __init__(
        self,
        featureName : str = "cartDist",
        dependencies : List[FeatureModule] = []
    ) -> None:
        
        self.computeCoords = len(dependencies) != 0
        if self.computeCoords and (len(dependencies) != 1 or dependencies[0].featureName != "cartCoords"):
            warnings.warn("Dependencies in CartesianDistances might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self, 
        pdb : AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Compute caresian distances
        if self.computeCoords:
            print("nöd ok")
            coords = torch.from_numpy(pdb.getCoordsets(0))
        else:
            coords = self.dependencies[0].data
        dist = torch.cdist(coords, coords).squeeze(0)

        return dist

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class SequenceDistances(FeatureModule):

    def __init__(
        self,
        featureName : str = "seqtDist"
    ) -> None:
        super().__init__(featureName)
    
    def forward(
        self, 
        pdb : AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Compute sequence distances
        x = torch.arange(pdb.numAtoms()).view(-1, 1).expand(pdb.numAtoms(), 2)
        return x[:, 0] - x[:, 1].view(-1, 1)

    def preFilter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class CloseNeighbours(FeatureModule):

    def __init__(
        self,
        threshold,
        featureName : str = "closeNeighbours",
        dependencies : List[FeatureModule] = [CartesianDistances()]
    ) -> None:
        self.threshold = threshold

        if len(dependencies) != 1:
            warnings.warn("Dependencies in CloseNeighbours might be errornous!", UserWarning)

        super().__init__(featureName, dependencies=dependencies)
    
    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return self.dependencies[0].data < self.threshold

    def preFilter(self, *args, **kwargs) -> bool:
        return all([ d.preFilter(*args, **kwargs) for d  in self.dependencies])


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
        return all([ d.preFilter(*args, **kwargs) for d  in self.dependencies])


class EdgeAttributes(FeatureModule):

    def __init__(
        self,
        featureName : str = "edgeAttributes",
        dependencies : List[FeatureModule] = [CartesianDistances()]
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
        return all([ d.preFilter(*args, **kwargs) for d  in self.dependencies])


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
        return all([ d.preFilter(*args, **kwargs) for d  in self.dependencies])


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
