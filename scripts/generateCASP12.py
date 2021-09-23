from pathlib import Path
from ProteinPairsGenerator.Data import ProteinNetDataset
import time

nCasp = 12

print("Finished constructing proteinNet{}".format(str(nCasp)))

t0 = time.time()

root = Path("proteinNetNew")
pro = root.joinpath("processed")

ProteinNetDataset(
    root = root,
    subsets = [pro.joinpath("testing"), pro.joinpath("validation"), pro.joinpath("training_100")],
    features = ProteinNetDataset.getGenericFeatures(),
    batchSize = 11000,
    caspVersion = nCasp
)

t1 = time.time()

print("Finished constructing proteinNet{} in {} seconds".format(str(nCasp), str(t1 - t0)))
