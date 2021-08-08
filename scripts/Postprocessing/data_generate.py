from pathlib import Path

from ProteinPairsGenerator.PostProcessing import SampleGeneratorStrokach
from ProteinPairsGenerator.StrokachModel import StrokachDataModule, StrokachModel

root = Path("proteinNetTesting")
subset = [root.joinpath("processed/testing")]
device = "cuda:2"


# Test Strokach
testerStrokach = SampleGeneratorStrokach("test.json", [{"maskFrac": 0.25}], 1, device)
dmStrokach = StrokachDataModule(root, subset, subset, subset)

print("Strokach alpha = 0.000")
mSalpha000 = StrokachModel.load_from_checkpoint("StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=21-step=902725.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerStrokach.run(mSalpha000, dmStrokach)
