from pathlib import Path
import sys

import torch

from ProteinPairsGenerator.Testing import TestProteinDesignIngrham, TestProteinDesignStrokach, TestProteinDesignJLo
from ProteinPairsGenerator.JLoModel import JLoDataModule, JLoModel
from ProteinPairsGenerator.StrokachModel import StrokachDataModule, StrokachModel
from ProteinPairsGenerator.IngrahamModel import IngrahamDataModule, IngrahamModel, IngrahamModelBERT
from ProteinPairsGenerator.BERTModel import TAPEWrapper

root = Path("proteinNetTesting")
subset = [root.joinpath("processed/testing")]
device = "cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")
args = ([{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 40, device, True, Path("results.json"))

# Test Strokach
testerIngraham = TestProteinDesignIngrham(*args)
dmIngraham = IngrahamDataModule(root, subset, subset, subset)

print("Ingraham alpha = 0.000")
mIalpha000 = IngrahamModel(vocab_input=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab_output=20)
mIalpha000.load_state_dict(torch.load("IngrahamStructural/Experiments/alpha=0/bestModel.ckpt", map_location=device))
testerIngraham.run(mIalpha000, dmIngraham)
del mIalpha000

# print("Ingraham alpha = 0.125")
# mIalpha125 = IngrahamModel(vocab_input=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab_output=20)
# mIalpha125.load_state_dict(torch.load("IngrahamStructural/Experiments/alpha=0.125/bestModel.ckpt", map_location=device))
# testerIngraham.run(mIalpha125, dmIngraham)
# del mIalpha125

# print("Ingraham BERT alpha = 0.000")
# mIalpha000 = IngrahamModelBERT(vocab_input=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab_output=20)
# mIalpha000.load_state_dict(torch.load("/mnt/ds3lab-scratch/jlokna/IngrahamBERT/Experiments/alpha=0/bestModel.ckpt", map_location=device))
# testerIngraham.run(mIalpha000, dmIngraham)
# del mIalpha000

testerIngraham.save_history()


# del dmIngraham
del testerIngraham


# # Test Strokach
# testerStrokach = TestProteinDesignStrokach([{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 40, device)
# dmStrokach = StrokachDataModule(root, subset, subset, subset)

# print("TAPE LM")
# mTAPE = TAPEWrapper()
# testerStrokach.run(mTAPE, dmStrokach)
# del mTAPE

# print("Strokach alpha = 0.000")
# mSalpha000 = StrokachModel.load_from_checkpoint("StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=21-step=902725.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
# testerStrokach.run(mSalpha000, dmStrokach)
# del mSalpha000

# print("Strokach alpha = 0.125")
# mSalpha125 = StrokachModel.load_from_checkpoint("StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.125_lr=0.0001/Checkpoints/epoch=34-step=1436154.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
# testerStrokach.run(mSalpha125, dmStrokach)
# del mSalpha125

# del dmStrokach
# del testerStrokach


# # Test BERT-Strokach
# testerJLo = TestProteinDesignJLo([{"maskFrac": 0.25}, {"maskFrac": 0.50}, {"maskFrac": 0.75}], 40, device)
# dmJLo = JLoDataModule(root, subset, subset, subset)

# print("JLo alpha = 0.000")
# mJalpha000 = JLoModel.load_from_checkpoint("StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=15-step=656527.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
# testerJLo.run(mJalpha000, dmJLo)
# del mJalpha000

# print("JLo alpha = 0.125")
# mJalpha125 = JLoModel.load_from_checkpoint("StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0.125_lr=0.0001/Checkpoints/epoch=20-step=861692.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
# testerJLo.run(mJalpha125, dmJLo)
# del mJalpha125

# del testerJLo
# del dmJLo
