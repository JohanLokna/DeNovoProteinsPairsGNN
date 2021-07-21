from ProteinPairsGenerator.Testing import TestProteinDesignJLo
from ProteinPairsGenerator.JLoModel import JLoDataModule, JLoModel

dm = JLoDataModule("proteinNetNew")
tester = TestProteinDesignJLo(dm, [{"maskFrac": 0.15},], 5)

m = JLoModel.load_from_checkpoint("StrokachJLo/Experiments/hidden_size=128_N=3_alpha=0_lr=0.0001/Checkpoints/epoch=34-step=3484249.ckpt", hidden_size=128, N=3, x_input_size=21, adj_input_size=2, output_size=20)

tester.run(m)
