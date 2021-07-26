import os
from multiprocessing import Pool
from typing import List, Optional, Union, IO

# Local imports
from .features import FeatureModule
from ProteinPairsGenerator.Data import GeneralData

# Colab dependent imports
if "get_ipython" in dir(os):
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

class DataGenerator:

    def __init__(
        self,
        features : List[FeatureModule],
        pool : Union[Pool, None] = None,
        batchSize : Optional[int] = None
    ) -> None:
        self.features = features
        self.batchSize = batchSize

        # Assert unique feature names
        assert len(set([f.featureName for f in self.features])) \
            == len(self.features)

        self.pool = pool

        # Might use in the future
        if not self.pool is None:
            raise NotImplementedError

    def addDataPoint(self, dataList, *args, **kwargs):

        # Iteratively construct features
        # Ensure that no errors        
        for f in self.features:
            if not f(*args, **kwargs):
                return

        # Append correctly computed features
        dataList.append(GeneralData(**{f.featureName: f.data for f in self.features if f.save}))

    def __call__(self, *args, **kwargs ):
        raise NotImplementedError


class DataGeneratorList(DataGenerator):
    
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        kwargsList : List,
    ):

        dataList = []
        for kwargs in tqdm(kwargsList):
            self.addDataPoint(dataList=dataList, **kwargs)

            if not self.batchSize is None and len(dataList) >= self.batchSize:
                yield dataList
                dataList = []
      
        if len(dataList) > 0:
            yield dataList


class DataGeneratorFile(DataGenerator):
    
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        inPath : IO,
    ):

        dataList = []
        inFile = open(str(inPath), 'r')

        count = 0
        while True:
            try:
                self.addDataPoint(dataList=dataList, inFile=inFile, iterationIdx = count)
                count += 1

            except EOFError:
                break

            if count % 1000 == 0:
                print("Completed {} samples".format(str(count)))

            if (not self.batchSize is None) and (len(dataList) >= self.batchSize):
                yield dataList
                dataList = []
      
        if len(dataList) > 0:
            yield dataList
