import os
from pathlib import Path
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Pytorch imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.seed import seed_everything
seed_everything(42)

# Local imports
from ProteinPairsGenerator.StrokachModel import StrokachModel, StrokachDataModule
from ProteinPairsGenerator.Data import ProteinNetDataset

# Set up dataset
root = Path("proteinNet")
pro = root.joinpath("processed")
nCasp = 12
dataset = ProteinNetDataset(
    root = root,
    subsets = [pro.joinpath("training_100"), pro.joinpath("validation"), pro.joinpath("testing")],
    features = [],
    batchSize = 11000,
    caspVersion = nCasp
)

num_features = 20
adj_input_size = 2
hidden_size = 128

model = StrokachModel(
    x_input_size=num_features + 1, 
    adj_input_size=adj_input_size, 
    hidden_size=hidden_size, 
    output_size=num_features,
    N=3
)

modelPath = Path("TestingModel")

mlflowPath = modelPath.joinpath("MLFlow")
mlflowPath.mkdir(parents=True, exist_ok=True)
mlf_logger = MLFlowLogger(
    experiment_name="Testing",
    tracking_uri="file:" + str(mlflowPath)
)

ckptPath = modelPath.joinpath("Checkpoints")
ckptPath.mkdir(parents=True, exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    dirpath=str(ckptPath),
    filename="{epoch}",
    save_top_k=3,
    monitor="valAcc",
    mode="max",
    verbose=True,
)

early_stop_callback = EarlyStopping(
   monitor='valAcc',
   min_delta=0.00,
   patience=5,
   verbose=False,
   mode='max'
)

dm = StrokachDataModule(dataset, pro.joinpath("training_100"), pro.joinpath("validation"), pro.joinpath("testing"))

trainer = pl.Trainer(
    gpus=1, 
    progress_bar_refresh_rate=500, 
    logger=mlf_logger,
    callbacks=[checkpoint_callback, early_stop_callback]
)
trainer.fit(model, dm)
