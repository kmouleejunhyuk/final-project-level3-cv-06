import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets.dataModule import CustomDataModule
from models.fasterrcnn_new_mAP import LitModel

from detection.config.detection_config import config as CONFIG

pl.seed_everything(CONFIG.seed)

def train_model(model_name, save_name=None, **kwargs):
    
    customDataModule = CustomDataModule(batch_size = CONFIG.batch_size, stage=CONFIG.stage)
    customDataModule.setup()

    train_loader = customDataModule.train_dataloader()
    val_loader = customDataModule.val_dataloader()

    if not save_name:
        save_name = model_name

    wandb_logger = WandbLogger(entity=CONFIG.entity, project=CONFIG.project, name=CONFIG.name)

    callbacks = []
    if CONFIG.EarlyStopping:
        callbacks.append(EarlyStopping(monitor="valid/val_mAP", patience=10, verbose=False, mode="max"))
    if CONFIG.ModelCheckpoint:
        callbacks.append(ModelCheckpoint(save_weights_only=True, mode='max', monitor='valid/val_mAP'))

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CONFIG.checkpoint_path, save_name),  
        gpus=CONFIG.gpus,
        precision=CONFIG.precision,
        max_epochs=CONFIG.epochs,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="valid/val_mAP", patience=10, verbose=False, mode="max"),
            ModelCheckpoint(save_weights_only=True, mode='max', monitor='valid/val_mAP') 
        ],  
    )  

    model = LitModel(config=CONFIG)
    trainer.fit(model, train_loader, val_loader)


train_model(
    model_name=CONFIG.model
)
