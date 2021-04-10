from pandas import DataFrame
from prody import fetchPDBviaHTTP, parsePDB
import time
from typing import List

def makeDatas(
    pdbDf : DataFrame, 
    verbose : bool = True,
    coordAtoms : List[str] = ['N', 'CA', 'C', 'O']
):

        data = []
        start = time.time()
        for i, (pdbName, _) in pdbDf.iterrows():

            entry = {}

            fetchPDBviaHTTP(pdbName, compressed=True)
            pdb = parsePDB(pdbName)

            entry["seq"] = pdb.ca.getSequence()
            entry["name"] = pdbName

            # Convert raw coords to np arrays
            for atom in coordAtoms:
                entry["coords"][atom] = pdb.select("name {}".format(atom)).getCoordsets().tolist()

            data.append(entry)

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                print('{} entries ({} loaded) in {:.1f} s'.format(len(data), i + 1, elapsed))
        
        return data

