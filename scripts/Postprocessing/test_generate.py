from pathlib import Path
import sys

import torch

from ProteinPairsGenerator.PostProcessing import SampleGeneratorStrokach, SampleGeneratorJLo, SampleGeneratorIngrham
from ProteinPairsGenerator.JLoModel import JLoDataModule, JLoModel
from ProteinPairsGenerator.StrokachModel import StrokachDataModule, StrokachModel
from ProteinPairsGenerator.IngrahamModel import IngrahamDataModule, IngrahamModelBERT, IngrahamModel

root = Path("/mnt/ds3lab-scratch/jlokna/proteinNetTesting")
trainSet = [root.joinpath("processed").joinpath("testing")]
device = "cuda:{}".format(sys.argv[1] if len(sys.argv) > 1 else "0")
args = (40, device)
basename = "guessesTest{}.json"

# Test Strokach
dmIngraham = IngrahamDataModule(root, trainSet=trainSet, valSet=trainSet, testSet=trainSet)

print("Ingraham alpha = 0.000")
tester = SampleGeneratorIngrham(basename.format("Ingraham000"), *args)
mIalpha000 = IngrahamModel(vocab_input=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab_output=20)
mIalpha000.load_state_dict(torch.load("/mnt/ds3lab-scratch/jlokna/IngrahamStructural/Experiments/alpha=0/bestModel.ckpt", map_location="cpu"))
tester.run(mIalpha000, dmIngraham.train_dataloader(), dmIngraham.transfer_batch_to_device, True)
del mIalpha000, tester

print("Ingraham alpha = 0.125")
tester = SampleGeneratorIngrham(basename.format("Ingraham125"), *args)
mIalpha125 = IngrahamModel(vocab_input=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab_output=20)
mIalpha125.load_state_dict(torch.load("/mnt/ds3lab-scratch/jlokna/IngrahamStructural/Experiments/alpha=0.125/bestModel.ckpt", map_location="cpu"))
tester.run(mIalpha125, dmIngraham.train_dataloader(), dmIngraham.transfer_batch_to_device, True)
del mIalpha125, tester

print("Ingraham BERT alpha = 0.000")
tester = SampleGeneratorIngrham(basename.format("IngrahamBERT000"), *args)
mIalpha000 = IngrahamModelBERT(vocab_input=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab_output=20)
mIalpha000.load_state_dict(torch.load("/mnt/ds3lab-scratch/jlokna/IngrahamBERT/Experiments/alpha=0/bestModel.ckpt", map_location="cpu"))
tester.run(mIalpha000, dmIngraham.train_dataloader(), dmIngraham.transfer_batch_to_device, True)
del mIalpha000, tester

del dmIngraham


# Test Strokach
dmStrokach = StrokachDataModule(root, trainSet=trainSet, valSet=trainSet, testSet=trainSet, num_workers=0, prefetch_factor=2, batch_size=1)

print("Strokach alpha = 0.000")
tester = SampleGeneratorStrokach(basename.format("Strokach000"), *args)
mSalpha000 = StrokachModel.load_from_checkpoint("/mnt/ds3lab-scratch/jlokna/StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=21-step=902725.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
tester.run(mSalpha000, dmStrokach.train_dataloader(), dmStrokach.transfer_batch_to_device, True)
del mSalpha000, tester

print("Strokach alpha = 0.125")
tester = SampleGeneratorStrokach(basename.format("Strokach125"), *args)
mSalpha125 = StrokachModel.load_from_checkpoint("/mnt/ds3lab-scratch/jlokna/StrokachExperiments/Experiments/hidden_size=128_N=3_alpha=0.125_lr=0.0001/Checkpoints/epoch=34-step=1436154.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
tester.run(mSalpha125, dmStrokach.train_dataloader(), dmStrokach.transfer_batch_to_device, True)
del mSalpha125, tester

del dmStrokach


# Test BERT-Strokach
dmJLo = JLoDataModule(root, trainSet=trainSet, valSet=trainSet, testSet=trainSet, num_workers=0, prefetch_factor=2, batch_size=1)

print("JLo alpha = 0.000")
tester = SampleGeneratorJLo(basename.format("JLo000"), *args)
mJalpha000 = JLoModel.load_from_checkpoint("/mnt/ds3lab-scratch/jlokna/StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0.0_lr=0.0001/Checkpoints/epoch=15-step=656527.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
tester.run(mJalpha000, dmJLo.train_dataloader(), dmJLo.transfer_batch_to_device, True)
del mJalpha000, tester

print("JLo alpha = 0.125")
tester = SampleGeneratorJLo(basename.format("JLo125"), *args)
mJalpha125 = JLoModel.load_from_checkpoint("/mnt/ds3lab-scratch/jlokna/StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0.125_lr=0.0001/Checkpoints/epoch=20-step=861692.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)
tester.run(mJalpha125, dmJLo.train_dataloader(), dmJLo.transfer_batch_to_device, True)
del mJalpha125, tester

del dmJLo
