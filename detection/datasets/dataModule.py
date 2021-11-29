from datasets.datasets import *
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size = 4):
        self.batch_size = batch_size
    
    def prepare_data(self):
        pass

    # https://wandb.ai/wandb_fc/korean/reports/Weights-Biases-Pytorch-Lightning---VmlldzozNzAxOTg
    def setup(self, stage=None):

        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "train" or stage == None:
            self.trainDataset = CustomDataset(annotation="/opt/ml/finalproject/data/train/resampled_errordel_train.json", 
                                                        data_dir="/opt/ml/finalproject/data/train/", mode="train", transforms=train_transform())
        if stage == "val" or stage == None:
            self.valDataset = CustomDataset(annotation="/opt/ml/finalproject/data/train/resampled_errordel_train.json", 
                                                        data_dir="/opt/ml/finalproject/data/train/", mode="val", transforms=valid_transform())
    
    def train_dataloader(self):
        return DataLoader(self.trainDataset, batch_size = self.batch_size, collate_fn=collate_fn, num_workers=2)
    def val_dataloader(self):
        return DataLoader(self.valDataset, batch_size = self.batch_size, collate_fn=collate_fn, num_workers=2)
    def test_dataloader(self):
        pass