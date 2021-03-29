import torch
from torch_geometric.data import Data

class PDBData(Data):

      def __init__(self, seq : torch.LongTensor, **kwargs):
          super().__init__(**kwargs)
          self.seq = seq
