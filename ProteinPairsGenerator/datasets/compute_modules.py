from prody import AtomGroup, ANM
from prody.atomic.select import Select
from typing import List, Mapping, Callable, Union, Generator, Any

import torch

from ProteinPairsGenerator.utils.amino_acids import seq_to_tensor, AMINO_ACIDS_MAP, \
                                                    CDRS_HEAVY, CDRS_LIGHT
from ProteinPairsGenerator.utils.cdr import getHeavyCDR, getLightCDR

# General purpose modules

class ComputeModule:

    def __init__(self, procs = None ) -> None:
        self.procs = procs

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def procs(self):
        return self.__procs

    @procs.setter
    def procs(self, var):
        self.__procs = var


class ComputeCombine(ComputeModule):

    def __init__(
        self, 
        testModules : List[ComputeModule], 
        aggr : Mapping[torch.Tensor, torch.Tensor]
    ) -> None:
        super().__init__()
        self.testModules = testModules
        self.aggr = aggr

    def __call__(self, *args, **kwargs) -> torch.Tensor:

        # Aggregate the results from the modules
        return self.aggr(torch.stack([test(*args, **kwargs) for test in self.testModules], dim=-1))


# Modules used for computing features

class GetSequence(ComputeModule):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(
        self, 
        pdb: AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:
        
        # Get sequence
        return seq_to_tensor(pdb.getSequence())


class GetSequenceCDR(ComputeModule):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(
        self,
        pdb: AtomGroup,
        Lchain: List[str] = [],
        Hchain: List[str] = [],
        *args,
        **kwargs
    ) -> torch.Tensor:
        
        # Get sequence
        seq = seq_to_tensor(pdb.getSequence())

        # Mask CDR in light chains in seq
        for c in Lchain:
          idx = Select().getIndices(pdb, "chain {}".format(c))
          for i, cdr in enumerate(getLightCDR(pdb.select("chain {}".format(c)).getSequence())):
            seq[idx[cdr]] = AMINO_ACIDS_MAP[CDRS_LIGHT[i]]

        # Mask CDR in heavy chains in seq
        for c in Hchain:
          idx = Select().getIndices(pdb, "chain {}".format(c))
          for i, cdr in enumerate(getHeavyCDR(pdb.select("chain {}".format(c)).getSequence())):
            seq[idx[cdr]] = AMINO_ACIDS_MAP[CDRS_HEAVY[i]]

        return seq


class GetModes(ComputeModule):

    def __init__(self, nModes : int) -> None:
        self.nModes = nModes
        super().__init__()

    def __call__(
        self, 
        pdb: AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # ANM set up mode calculations
        pdbANM = ANM(pdb)
        pdbANM.buildHessian(pdb)
        pdbANM.calcModes(self.nModes)

        # Make into array and reshape to [numAtoms, -1]
        modes = torch.from_numpy(pdbANM.getArray()).type(torch.FloatTensor)
        return modes.view(pdb.numAtoms(), -1)


class GetCartesianDistances(ComputeModule):

    def __init__(self):
        super().__init__()
    
    def __call__(
        self, 
        pdb: AtomGroup,
        *args,
        **kwargs
    ) -> torch.FloatTensor:

        # Compute caresian distances
        coords = torch.from_numpy(pdb.getCoordsets())
        dist = torch.cdist(coords, coords).squeeze(0).type(torch.FloatTensor)

        # try:
        #     assert len(dist.shape) == 2
        # except Exception:
        #     print(dist.shape)
        #     print(coords.shape)
        #     print(pdb.getTitle())

        return dist


class GetSequenceDistances(ComputeModule):

    def __init__(self):
        super().__init__()
    
    def __call__(
        self, 
        pdb: AtomGroup,
        *args,
        **kwargs
    ) -> torch.FloatTensor:

        # Compute sequence distances
        x = torch.arange(pdb.numAtoms()).view(-1, 1).expand(pdb.numAtoms(), 2)
        return x[:, 0] - x[:, 1].view(-1, 1)


# Modules used for testing

class TestChainsPresent(ComputeModule):

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, 
        pdb: AtomGroup, 
        Lchains: List[str] = [],
        Hchains: List[str] = [],
        *args, 
        **kwargs
    ) -> torch.BoolTensor:

        # Test that the L and H chains are contained 
        # in the set of chains in pdb
        return torch.BoolTensor(set(Lchains + Hchains) <= set(pdb.getChids()))


class TestUpperBound(ComputeModule):
    
    def __init__(self, threshold, dim: Union[int, type(None)] = None,):
        super().__init__()
        self.threshold = threshold
        self.dim = dim

    def __call__(
        self, 
        pdb: AtomGroup, 
        edgeAttr: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.BoolTensor:

        # Test if bellow theshold
        if self.dim is None:
            return edgeAttr < self.threshold
        else:
            return edgeAttr[:, :, self.dim] < self.threshold
