from pathlib import Path
import sys

from ProteinPairsGenerator.Testing import TestProteinDesignIngrham, TestProteinDesignStrokach, TestProteinDesignJLo
from ProteinPairsGenerator.JLoModel import JLoDataModule, JLoModel
from ProteinPairsGenerator.StrokachModel import StrokachDataModule, StrokachModel
from ProteinPairsGenerator.IngrahamModel import IngrahamDataModule, IngrahamModel
from ProteinPairsGenerator.PostProcessing import CorrectorFullSoftBERT

import torch

root = Path("proteinNetNew")
subset = [root.joinpath("processed/validation")]
device = "cpu" #"cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")

# Corrector
corrector = CorrectorFullSoftBERT.load_from_checkpoint("SoftCorrectorV2/Experiments/hidden_size=256_N=2_alpha=0.8_lr=0.0001_dropout=0.3/Checkpoints/last.ckpt", hidden_size=256, N=2, dropout=0.3, alpha=0.8, lr=1e-4)

# Test Strokach
testerIngraham = TestProteinDesignIngrham([{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 4, device)
dmIngraham = IngrahamDataModule(root, subset, subset, subset)

print("Ingraham alpha = 0.000")
mIalpha000 = IngrahamModel(vocab_input=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab_output=20)
mIalpha000.load_state_dict(torch.load("IngrahamStructural/Experiments/alpha=0/bestModel.ckpt", map_location=device))
testerIngraham.run(mIalpha000, dmIngraham, corrector=corrector, classDim=-1)
del mIalpha000

print("Ingraham alpha = 0.125")
mIalpha125 = IngrahamModel(vocab_input=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab_output=20)
mIalpha125.load_state_dict(torch.load("IngrahamStructural/Experiments/alpha=0.125/bestModel.ckpt", map_location=device))
testerIngraham.run(mIalpha125, dmIngraham, corrector=corrector, classDim=-1)
del mIalpha125

del dmIngraham
del testerIngraham


# Test Strokach
testerStrokach = TestProteinDesignStrokach([{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 4, device)
dmStrokach = StrokachDataModule(root, subset, subset, subset)

print("Strokach alpha = 0.000")
mSalpha000 = StrokachModel.load_from_checkpoint("StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=21-step=902725.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerStrokach.run(mSalpha000, dmStrokach, corrector=corrector)
del mSalpha000

print("Strokach alpha = 0.125")
mSalpha125 = StrokachModel.load_from_checkpoint("StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.125_lr=0.0001/Checkpoints/epoch=34-step=1436154.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerStrokach.run(mSalpha125, dmStrokach, corrector=corrector)
del mSalpha125

del dmStrokach
del testerStrokach


# Test BERT-Strokach
testerJLo = TestProteinDesignJLo([{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 4, device)
dmJLo = JLoDataModule(root, subset, subset, subset)

print("JLo alpha = 0.000")
mJalpha000 = JLoModel.load_from_checkpoint("StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=15-step=656527.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerJLo.run(mJalpha000, dmJLo, corrector=corrector)
del mJalpha000

print("JLo alpha = 0.125")
mJalpha125 = JLoModel.load_from_checkpoint("StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0.125_lr=0.0001/Checkpoints/epoch=20-step=861692.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
testerJLo.run(mJalpha125, dmJLo, corrector=corrector)
del mJalpha125

del testerJLo
del dmJLo
