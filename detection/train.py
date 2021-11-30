import os
import torch
import torch.nn as nn
from models.models import LitModel
from datasets.dataModule import CustomDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/opt/ml/finalproject/saved_models/"

# Function for setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
        gpus=1 if str(device) == "cuda:0" else 0,
        # How many epochs to train for if no patience is set

        max_epochs=180,
        # callbacks=[
        #     ModelCheckpoint(
        #         save_weights_only=True, mode="min", monitor="loss"
        #     ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
        #     LearningRateMonitor("epoch"),
        # ],  # Log learning rate every epoch
        # progress_bar_refresh_rate=1
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
        
    pl.seed_everything(42)  # To be reproducable
    model = LitModel(model_name=model_name, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    # model = LitModel.load_from_checkpoint(
    #     trainer.checkpoint_callback.best_model_path
    # )  # Load best checkpoint after training

    # Test best model on validation and test set
    # val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    # # test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    # test_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    
    # result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    # result = {"test": 0, "val": val_result["val_iou"]}
    # return result

train_model(
    model_name="fcnn",
    model_hparams={},
    optimizer_name="SGD",
    optimizer_hparams={"lr": 0.0001, "momentum": 0.9, "weight_decay": 1e-4},
)