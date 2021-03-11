import pandas as pd
from pathlib import Path
from prody import AtomGroup
from prody.atomic.select import Select
from typing import List, Mapping

import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from ProteinPairsGenerator.utils import *
from .pdb_dataset import PDBInMemoryDataset
from .utils import transform_edge_attr


def cdr_extracter(data_pdb: AtomGroup, Lchain: List[str] = [], Hchain: List[str] = []) -> Data:

    # Only use alpha C atom for each residue
    set_pdb = data_pdb.select("name CA")
    
    # Get sequence
    seq = seq_to_tensor(set_pdb.getSequence())

    # Mask CDR in light chains
    for c in Lchain + Hchain:
      idx = Select().getIndices(set_pdb, "chain {}".format(c))
      for cdr in getLightCDR(set_pdb.select("chain {}".format(c)).getSequence()):
        seq[idx[cdr]] = AMINO_ACIDS_MAP[AMINO_ACID_NULL]

    # Mask CDR in heavy chains
    for c in Hchain:
      idx = Select().getIndices(set_pdb, "chain {}".format(c))
      for cdr in getHeavyCDR(set_pdb.select("chain {}".format(c)).getSequence()):
        seq[idx[cdr]] = AMINO_ACIDS_MAP[AMINO_ACID_NULL]

    # Find intersequence distance
    n = seq.shape[0]

    ids = torch.arange(n, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
    seq_distances = torch.cdist(ids, ids).flatten()

    # Compute caresian distances
    coords = torch.from_numpy(set_pdb.getCoordsets())
    cart_distances = torch.cdist(coords, coords).squeeze(0)

    # Compute edges and their atributes
    mask = cart_distances < 12
    edge_attr = torch.stack(
      [cart_distances.flatten(), seq_distances.flatten()], dim=1
    )
    edge_attr = edge_attr[mask.flatten(), :]
    edge_index = torch.stack(torch.where(mask), dim=0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    # Create data point
    data = Data(x=seq, edge_index=edge_index, edge_attr=edge_attr)
    data = transform_edge_attr(data)
    data = data.coalesce()

    # Assertions
    assert not data.contains_self_loops()
    assert data.is_coalesced()

    return data


class SAbDabInMemoryDataset(PDBInMemoryDataset):

    def __init__(
        self, 
        summary_file : Path,
        root : Path,
        pre_transform : Mapping[AtomGroup, Data] = cdr_extracter,
        **kwargs
    ) -> None:

        fields = ["pdb", "Hchain", "Lchain"]
        summary = pd.read_csv(summary_file, usecols=fields, delimiter="\t")

        concat = lambda x: x.dropna().tolist()
        summary = summary.groupby(summary["pdb"]).agg({"Hchain": concat, "Lchain": concat})

        super().__init__(root=root, pdbs=summary, pre_transform=pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()
