from pl_examples import cli_lightning_logo
from pytorch_lightning.utilities.cli import LightningCLI

from ProteinPairsGenerator.StrokachModel import StrokachModel, StrokachDataModule
from ProteinPairsGenerator.DistilationKnowledge import getKDModel

def cli_main():

    print("1")
    cli = LightningCLI(
        getKDModel(StrokachModel, 0.5),
        StrokachDataModule
    )
    print("2")
    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()
