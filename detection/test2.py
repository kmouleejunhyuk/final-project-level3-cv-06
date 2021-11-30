import os

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import LightningModule
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from datasets.dataModule import CustomDataModule
from models.efficientdet import WheatModel

pl.seed_everything(42)

customDataModule = CustomDataModule(batch_size=8)
customDataModule.setup()

train_loader = customDataModule.train_dataloader()
val_loader = customDataModule.val_dataloader()

model = WheatModel()

trainer = pl.Trainer(gpus=1, precision=16, max_epochs=10)
trainer.fit(model, train_loader, val_loader)

