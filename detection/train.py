import os

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from datasets.dataModule import CustomDataModule
from models.fasterrcnn import LitModel


# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/opt/ml/finalproject/saved_models/"
# Function for setting the seed
pl.seed_everything(42)


def train_model(model_name, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    customDataModule = CustomDataModule(batch_size = 4)
    customDataModule.setup()

    train_loader = customDataModule.train_dataloader()
    val_loader = customDataModule.val_dataloader()

    if save_name is None:
        save_name = model_name

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
        # We run on a single GPU (if possible)
        gpus=1,
        # How many epochs to train for if no patience is set
        max_epochs=180,
        callbacks=[
            EarlyStopping(monitor="total_mAP", patience=5, verbose=False, mode="max")
             # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
        ],  # Log learning rate every epoch
        # progress_bar_refresh_rate=1
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
        
    pl.seed_everything(42)  # To be reproducable
    model = LitModel()
    trainer.fit(model, train_loader, val_loader)


train_model(
    model_name="fcnn",
    model_hparams={},
    optimizer_name="SGD",
    optimizer_hparams={"lr": 0.0001, "momentum": 0.9, "weight_decay": 1e-4},
)