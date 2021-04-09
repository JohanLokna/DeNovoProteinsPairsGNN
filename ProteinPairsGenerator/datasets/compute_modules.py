from pathlib import Path
from prody import AtomGroup, ANM
from prody.atomic.select import Select
from typing import List, Mapping, Callable, Union, Generator, Any

import torch

from ProteinPairsGenerator.utils.amino_acids import seq_to_tensor, \
                                                    AMINO_ACIDS_MAP, AMINO_ACIDS_BASE, \
                                                    CDRS_HEAVY, CDRS_LIGHT, \
                                                    CHAIN_NULL, CHAIN_HEAVY, CHAIN_LIGHT, CHAIN_ANTIGEN, \
                                                    CHAINS_MAP
from ProteinPairsGenerator.utils.cdr import getHeavyCDR, getLightCDR

# General purpose modules

class ComputeModule:

    def __init__(self, filename : Path, featureName : str) -> None:
        self.filename = filename
        self.featureName = featureName
        self.data = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def pre_filter(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    @property
    def data(self):
        return self.data_

    @data.setter
    def data(self, value):
        self.data_ = value

    def save(self, filename : Union[Path, None] = None):
        torch.save({self.featureName: self.data}, self.filename if filename is None else filename)

    def __call__(self, argList : List):
        self.data = {i : self.forward(**x) for i, x in enumerate(argList)}

# Modules used for computing features

class GetSequence(ComputeModule):

    def __init__(
        self,
        filename : Path = Path("./seq.pt"), 
        featureName : str = "seq"
    ) -> None:
        super().__init__(filename, featureName)
    
    def forward(
        self, 
        pdb: AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:
        
        # Get sequence
        return seq_to_tensor(pdb.getSequence())

    def pre_filter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class GetSequenceCDR(ComputeModule):

    def __init__(
        self,
        filename : Path = Path("./seqCDR.pt"),
        featureName : str = "seqCDR",
        hmmerpath : str = "/usr/bin/",
    ) -> None:
        self.hmmerpath = hmmerpath
        super().__init__(filename, featureName)
    
    def forward(
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
          for i, cdr in enumerate(getLightCDR(pdb.select("chain {}".format(c)).getSequence(), hmmerpath=self.hmmerpath)):
            seq[idx[cdr]] = AMINO_ACIDS_MAP[CDRS_LIGHT[i]]

        # Mask CDR in heavy chains in seq
        for c in Hchain:
          idx = Select().getIndices(pdb, "chain {}".format(c))
          for i, cdr in enumerate(getHeavyCDR(pdb.select("chain {}".format(c)).getSequence(), hmmerpath=self.hmmerpath)):
            seq[idx[cdr]] = AMINO_ACIDS_MAP[CDRS_HEAVY[i]]

        return seq

    def pre_filter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs \
           and "Lchain" in kwargs \
           and "Hchain" in kwargs \
           and set(kwargs["Lchain"] + kwargs["Hchain"]) <= set(kwargs["pdb"].getChids()) \
           and set(kwargs["pdb"].getSequence()) <= set(AMINO_ACIDS_BASE)


class GetChainsDescription(ComputeModule):

    def __init__(
        self,
        filename : Path = Path("./seqChains.pt"),
        featureName : str = "seqChains"
    ) -> None:
        super().__init__(filename, featureName)
    
    def forward(
        self,
        pdb: AtomGroup,
        Lchain: List[str] = [],
        Hchain: List[str] = [],
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

    def pre_filter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs \
           and "Lchain" in kwargs \
           and "Hchain" in kwargs \
           and set(kwargs["Lchain"] + kwargs["Hchain"]) <= set(kwargs["pdb"].getChids())


class GetModes(ComputeModule):

    def __init__(
        self,
        filename : Path = Path("./modes.pt"),
        featureName : str = "modes",
        nModes : int = 20
    ) -> None:
        self.nModes = nModes
        super().__init__(filename, featureName)

    def forward(
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
        modes = torch.from_numpy(pdbANM.getArray())
        return modes.view(pdb.numAtoms(), -1)

    def pre_filter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class GetCartesianDistances(ComputeModule):

    def __init__(
        self,
        filename : Path = Path("./cartDist.pt"),
        featureName : str = "cartDist"
    ) -> None:
        super().__init__(filename, featureName)
    
    def forward(
        self, 
        pdb: AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Compute caresian distances
        coords = torch.from_numpy(pdb.getCoordsets())
        dist = torch.cdist(coords, coords).squeeze(0)

        return dist

    def pre_filter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs


class GetSequenceDistances(ComputeModule):

    def __init__(
        self,
        filename : Path = Path("./seqtDist.pt"),
        featureName : str = "seqtDist"
    ) -> None:
        super().__init__(filename, featureName)
    
    def forward(
        self, 
        pdb: AtomGroup,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # Compute sequence distances
        x = torch.arange(pdb.numAtoms()).view(-1, 1).expand(pdb.numAtoms(), 2)
        return x[:, 0] - x[:, 1].view(-1, 1)

    def pre_filter(self, *args, **kwargs) -> bool:
        return "pdb" in kwargs
