
import editdistance
import pandas as pd
from pathlib import Path
from prody import AtomGroup
from prody.atomic.select import Select
from typing import List, Mapping, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from ProteinPairsGenerator.utils import *
from .pdb_dataset import PDBInMemoryDataset
from .utils import transform_edge_attr, splitDistinctSequences


def dist(x, y):
    AMINO_ACIDS_MAP[AMINO_ACID_NULL]
    cdrX = x.x[x.x == AMINO_ACIDS_MAP[AMINO_ACID_NULL]]
    cdrY = y.x[y.x == AMINO_ACIDS_MAP[AMINO_ACID_NULL]]
    return editdistance.eval(cdrX, cdrY)


def cdrExtracter(data_pdb: AtomGroup, Lchain: List[str] = [], Hchain: List[str] = []) -> Data:

    try:
        # Only use alpha C atom for each residue
        set_pdb = data_pdb.select("name CA")
        
        # Get sequence
        seq = seq_to_tensor(set_pdb.getSequence())
        y = seq.clone().detach()

        # Only valid sequences are accepted
        assert not (y == 20).any()

        # Mask CDR in light chains
        for c in Lchain:
          idx = Select().getIndices(set_pdb, "chain {}".format(c))
          for cdr in getLightCDR(set_pdb.select("chain {}".format(c)).getSequence()):
            seq[idx[cdr]] = AMINO_ACIDS_MAP[AMINO_ACID_NULL]


        # Mask CDR in heavy chains
        for c in Hchain:
          idx = Select().getIndices(set_pdb, "chain {}".format(c))
          for cdr in getHeavyCDR(set_pdb.select("chain {}".format(c)).getSequence()):
            seq[idx[cdr]] = AMINO_ACIDS_MAP[AMINO_ACID_NULL]

        # Compute caresian distances
        coords = torch.from_numpy(set_pdb.getCoordsets())
        cart_distances = torch.cdist(coords, coords).squeeze(0)

        # Compute edges and their atributes
        mask = cart_distances < 12
        edge_attr = cart_distances.flatten()[mask.flatten()].unsqueeze(-1)
        edge_index = torch.stack(torch.where(mask), dim=0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        # Create data point
        seq = seq.type(torch.LongTensor)
        edge_attr = edge_attr.type(torch.FloatTensor)
        edge_index = edge_index.type(torch.LongTensor)

        data = Data(x=seq, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data = transform_edge_attr(data)
        data = data.coalesce()

        # Assertions
        assert not data.contains_self_loops()
        assert data.is_coalesced()

        return data
    
    except Exception:
        return


class SAbDabInMemoryDataset(PDBInMemoryDataset):

    def __init__(
        self, 
        summary_file : Path,
        root : Path,
        pre_transform : Mapping[AtomGroup, Data] = cdrExtracter,
        splitter : Mapping[Union[Dataset, List[float]], List[List[int]]] = splitDistinctSequences(dist, 15),
        **kwargs
    ) -> None:

        fields = ["pdb", "Hchain", "Lchain"]
        summary = pd.read_csv(summary_file, usecols=fields, delimiter="\t")

        concat = lambda x: x.dropna().tolist()
        summary = summary.groupby(summary["pdb"]).agg({"Hchain": concat, "Lchain": concat})

        super().__init__(root=root, pdbs=summary.head(3000), 
                         pre_transform=pre_transform, splitter=splitter,
                         **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()
