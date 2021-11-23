from datasets import *
from pytorch_lightning import LightningDataModule

class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size = 4):
        self.batch_size = batch_size
    
    def prepare_data(self):
        pass

    # https://wandb.ai/wandb_fc/korean/reports/Weights-Biases-Pytorch-Lightning---VmlldzozNzAxOTg
    def setup(self, stage=None):
        if stage == "train" or stage == None:
            self.trainDataset = CustomDataset(annotation="/opt/ml/finalproject/data/train/modified_train_dummy.json", 
                                                        data_dir="/opt/ml/finalproject/data/train/", mode=self.mode, transforms=train_transform())
        if stage == "val" or stage == None:
            self.valDataset = CustomDataset(annotation="/opt/ml/finalproject/data/eval/modified_eval_dummy.json", 
                                                        data_dir="/opt/ml/finalproject/data/eval/", mode=self.mode, transforms=valid_transform())
    
    def train_dataloader(self):
        return DataLoader(self.trainDataset, batch_size = self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.valDataset, batch_size = self.batch_size)
    def test_dataloader(self):
        pass