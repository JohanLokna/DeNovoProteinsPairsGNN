from typing import Union

import torch
from torch_geometric.data import Data

class PDBData(Data):

      def __init__(
          self, 
          seq : Union[torch.LongTensor, None] = None, 
          meta : Union[dict] = {},
          **kwargs
      ) -> None:
          super().__init__(**kwargs)
          self.seq = seq
          self.meta = meta
