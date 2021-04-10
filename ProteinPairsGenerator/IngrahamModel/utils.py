from pandas import DataFrame
import time
from tqdm.notebook import tqdm
from typing import List
from .util_mmtf import *

def makeData(
    pdbDf : DataFrame, 
    verbose : bool = True,
    coordAtoms : List[str] = ['N', 'CA', 'C', 'O']
):

        data = []
        start = time.time()
        for i, (pdb, metaData) in enumerate(tqdm(pdbDf.iterrows())):

            for chain in metaData["Hchain"] + metaData["Lchain"] + metaData["antigen_chain"]:
                try:

                    chain_dict = mmtf_parse(pdb, chain)
                    chain_name = pdb + '.' + chain
                    chain_dict['name'] = chain_name
                    dataset.append(chain_dict)

                except Exception as e:
                    print(e)

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                print('{} entries ({} loaded) in {:.1f} s'.format(len(data), i + 1, elapsed))
        
        return data
