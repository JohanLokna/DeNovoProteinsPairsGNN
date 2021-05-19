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
from ProteinPairsGenerator.IngrahamModel import IngrahamModel, IngrahamDataModule
from ProteinPairsGenerator.Data import ProteinNetDataset

# Set up dataset
root = Path("proteinNetNew")
pro = root.joinpath("processed")
nCasp = 12
dataset = ProteinNetDataset(
    root = root,
    subsets = [pro.joinpath("training_100"), pro.joinpath("validation"), pro.joinpath("testing")],
    features = ProteinNetDataset.getGenericFeatures(),
    batchSize = 11000,
    caspVersion = nCasp
)

# Set up model
model = IngrahamModel(
    in_size=21, 
    out_size=20,
    node_features=128,
    edge_features=128, 
    hidden_dim=128,
    k_neighbors=30,
    protein_features="full",
    dropout=0.1,
    use_mpnn=False
)

# Set up logging and checkpoints
modelPath = Path("TestingModel")

mlflowPath = modelPath.joinpath("MLFlow")
mlflowPath.mkdir(parents=True, exist_ok=True)
mlf_logger = MLFlowLogger(
    experiment_name="Ingraham",
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

# Set up stopping
early_stop_callback = EarlyStopping(
   monitor='valAcc',
   min_delta=0.00,
   patience=5,
   verbose=False,
   mode='max'
)

# Set up datamodule
dm = IngrahamDataModule(
      dataset,
      pro.joinpath("validation"), 
      pro.joinpath("validation"), 
      pro.joinpath("testing"),
      batchSize=4
)

# Train
trainer = pl.Trainer(
    gpus=1, 
    progress_bar_refresh_rate=10000, 
    logger=mlf_logger,
    callbacks=[checkpoint_callback, early_stop_callback]
)
trainer.fit(model, dm)
