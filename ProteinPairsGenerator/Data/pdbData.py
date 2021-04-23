from typing import Union, Dict

from torch_geometric.data import Data

class PDBData(Data):

      def __init__(
          self,
          **kwargs
      ) -> None:
          self.__dict__.update(**kwargs)
          super().__init__(**kwargs)
