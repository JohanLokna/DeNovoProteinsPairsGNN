import math
from multiprocessing import Pool
from typing import List, Union, Generator

# Local imports
from .features import FeatureModule
from ProteinPairsGenerator.Data import PDBData

# Colab dependent imports
if 'google.colab' in str(get_ipython()):
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

class DataGenerator:

    def __init__(
        self,
        features : List[FeatureModule],
        batchSize : Union[int, None] = None,
        pool : Union[Pool, None] = None,

    ) -> None:
        self.features = features
        self.batchSize = batchSize
        self.pool = pool
  
    def getNumBatches(
        self,
        size : int
    ) -> int:

        # If no batch size, use all blocks
        batchSize = len(kwargsList) if self.batchSize is None else self.batchSize

        return math.ceil(size / batchSize)

    def __call__(
        self,
        kwargsList : List,
    ) -> Generator[List[PDBData], None, None]:

        # Assert unique feature names
        assert len(set([f.featureName for f in self.features])) \
            == len(self.features)

        # If no batch size, use all blocks
        batchSize = len(kwargsList) if self.batchSize is None else self.batchSize

        # Might use 
        if not self.pool is None:
            raise NotImplementedError

        dataList = []
        for kwargs in tqdm(kwargsList):

            # Iteratively construct features
            # Ensure that no errors
            error = False
            
            for f in self.features:
                if not f(**kwargs):
                    error = True
                    break
            if error:
                continue

            # Append correctly computed features
            dataList.append(PDBData(**{f.featureName: f.data for f in self.features}))

            # Write batch
            if len(dataList) >= batchSize:
                yield dataList
                dataList = []

        # Wirte last batch
        if len(dataList) > 0:
           yield dataList
