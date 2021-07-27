from pathlib import Path
import sys

from ProteinPairsGenerator.Testing import TestProteinDesignJLo, TestProteinDesignStrokach
from ProteinPairsGenerator.JLoModel import JLoDataModule, JLoModel
from ProteinPairsGenerator.StrokachModel import StrokachDataModule, StrokachModel

root = Path("proteinNetTesting")
subset = [root.joinpath("processed/testing")]
device = "cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")

# Test Strokach

dmStrokach = StrokachDataModule(root, subset, subset, subset)
testerStrokach = TestProteinDesignStrokach(dmStrokach, [{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 5, device)

print("Strokach alpha = 0.000")
mSalpha000 = StrokachModel.load_from_checkpoint("StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=21-step=902725.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerStrokach.run(mSalpha000)
del mSalpha000

print("Strokach alpha = 0.125")
mSalpha125 = StrokachModel.load_from_checkpoint("StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.125_lr=0.0001/Checkpoints/epoch=34-step=1436154.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerStrokach.run(mSalpha125)
del mSalpha125

del testerStrokach
del dmStrokach

# Test Strokach with embedding

dmJLo = JLoDataModule(root, subset, subset, subset)
testerJLo = TestProteinDesignJLo(dmJLo, [{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 5, device)

print("JLo alpha = 0.000")
mJalpha000 = JLoModel.load_from_checkpoint("StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=15-step=656527.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerJLo.run(mJalpha000)
del mJalpha000

print("JLo alpha = 0.000")
mJalpha125 = JLoModel.load_from_checkpoint("StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0.125_lr=0.0001/Checkpoints/epoch=20-step=861692.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerJLo.run(mJalpha125)
del mJalpha125

del testerJLo
del dmJLo
