import math
from multiprocessing import Pool
from typing import List, Union, IO

# Local imports
from .features import FeatureModule
from ProteinPairsGenerator.Data import GeneralData

# Colab dependent imports
if 'google.colab' in str(get_ipython()):
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

class DataGenerator:

    def __init__(
        self,
        features : List[FeatureModule],
        pool : Union[Pool, None] = None
    ) -> None:
        self.features = features

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
        dataList.append(GeneralData(**{f.featureName: f.data for f in self.features}))

    def __call__(self, *args, **kwargs ) -> List[GeneralData]:
        raise NotImplementedError


class DataGeneratorList(DataGenerator):
    
    def __init__(
        self,
        features : List[FeatureModule],
        pool : Union[Pool, None] = None
    ) -> None:
        super().__init__(features, pool)

    def __call__(
        self,
        kwargsList : List,
    ) -> List[GeneralData]:

        dataList = []
        for kwargs in tqdm(kwargsList):
            self.addDataPoint(dataList=dataList, **kwargs)
            
        return dataList


class DataGeneratorFile(DataGenerator):
    
    def __init__(
        self,
        features : List[FeatureModule],
        pool : Union[Pool, None] = None
    ) -> None:
        super().__init__(features, pool)

    def __call__(
        self,
        inPath : IO,
    ) -> List[GeneralData]:

        dataList = []
        inFile = open(str(inPath), 'r')
        while True:
            try:
                self.addDataPoint(dataList=dataList, inFile=inFile)
                
                if len(dataList) == 100:
                    return dataList
            except EOFError:
                break
      
        return dataList
