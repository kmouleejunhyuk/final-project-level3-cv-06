from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
#from models.efficientdet import efficientdet
from models.fasterrcnn import FCNN, _evaluate_iou

#model_dict = {"effi" : efficientdet}
act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}

# def create_model(model_name, model_hparams):
#     if model_name in model_dict:
#         return model_dict[model_name](**model_hparams)
#     else:
#         assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'

# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
class LitModel(LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """        
        super().__init__()
        # argument로 제공된 것들 self.hparams내의 속성으로 저장
        self.save_hyperparameters()
        self.model = FCNN()
        self.loss_module = nn.CrossEntropyLoss()

        # self.example_input_array = torch.zeros((1, 3, 1500, 1500), dtype=torch.float32)
    def forward(self, imgs, target):
        # Forward function that is run when visualizing the graph
        return self.model(imgs, target)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
        
        # 현재 필요 X
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, target = batch
        loss_dict = self.forward(imgs, target)
        losses = sum(loss for loss in loss_dict.values())
        #print('training_step loss_dict', loss_dict)
        self.log("loss_classifier", loss_dict['loss_classifier'])
        self.log("loss_box_reg", loss_dict['loss_box_reg'])
        self.log("loss_objectness", loss_dict['loss_objectness'])
        self.log("loss_rpn_box_reg", loss_dict['loss_rpn_box_reg'])

        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, target = batch
        outs = self.model(imgs, target)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(target, outs)]).mean()

        self.log("val_iou", iou)
        return {"val_iou": iou}
    

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)   


