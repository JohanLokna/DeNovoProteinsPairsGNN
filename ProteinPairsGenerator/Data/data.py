from typing import Union, Dict

from torch_geometric.data import Data

class GeneralData(Data):

      def __init__(
          self,
          **kwargs
      ) -> None:
          self.__dict__.update(**kwargs)
          super().__init__(**kwargs)
