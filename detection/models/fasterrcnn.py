import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import LightningModule
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou

from models.metrics import mean_average_precision


def _evaluate_iou(target, pred):
    """Evaluate intersection over union (IOU) for target from dataset and output prediction from model."""

    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

class LitModel(LightningModule):
    def __init__(self):
        super().__init__()
        num_classes = 39 # include background (0: background)
        # self.backbone = torchvision.models.resnet50(pretrained=True)
        # del self.backbone.fc
        # self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # self.backbone.out_channels = 1280

        # self.anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 256),),
        #                                         aspect_ratios=((0.5, 1.0, 2.0),))
        # self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        #                                                      output_size=7,
        #                                                      sampling_ratio=2)
        # self.model = FasterRCNN(backbone=self.backbone,
        #                         num_classes=num_classes,
        #                         rpn_anchor_generator=self.anchor_generator,
        #                         box_roi_pool=self.roi_pooler)
        self.model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
        
        self.class_aps = {str(i):[] for i in range(39)}

    def forward(self, imgs):
        self.model.eval()
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, targets = batch
        loss_dict = self.model(imgs, targets) # loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "loss_classifier": loss_dict['loss_classifier'], "loss_box_reg": loss_dict['loss_box_reg'],
                "loss_objectness": loss_dict['loss_objectness'], "loss_rpn_box_reg": loss_dict['loss_rpn_box_reg'],  "log": loss_dict}
    
    def training_epoch_end(self, outs):
        loss_sum = torch.stack([o["loss"] for o in outs]).sum()
        loss_classifier = torch.stack([o["loss_classifier"] for o in outs]).mean()
        loss_box_reg = torch.stack([o["loss_box_reg"] for o in outs]).mean()
        loss_objectness = torch.stack([o["loss_objectness"] for o in outs]).mean()
        loss_rpn_box_reg = torch.stack([o["loss_rpn_box_reg"] for o in outs]).mean()
        self.log('Train/loss_sum', loss_sum)
        self.log('Train/loss_classifier', loss_classifier)
        self.log('Train/loss_box_reg', loss_box_reg)
        self.log('Train/loss_objectness', loss_objectness)
        self.log('Train/loss_rpn_box_reg', loss_rpn_box_reg)

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        outs = self.model(imgs)

        step_mAP = 0.
        for j,o in enumerate(outs):
            pred_boxes = o['boxes']
            pred_labels = o['labels']
            pred_scores = o['scores']
            pred_idx = torch.tensor([i for i in range(len(pred_labels))], device=pred_labels.device)

            target_boxes = targets[j]['boxes']
            target_labels = targets[j]['labels']
            target_idx = torch.tensor([i for i in range(len(target_labels))], device=target_labels.device) 

            iou_thres = 0.5
            mAP, APs = mean_average_precision(pred_idx, pred_scores, pred_labels, pred_boxes, target_idx, target_labels, target_boxes, iou_thres)
            for k,v in APs.items():
                if v != -1.:
                    self.class_aps[k].append(v)
            step_mAP += mAP.float()
        step_mAP /= len(outs)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou, "val_mAP": step_mAP}

    def validation_epoch_end(self, outs):
        total_mAP = torch.stack([o["val_mAP"] for o in outs]).mean()
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()

        logs = {"val_iou": avg_iou}
        self.log("valid/val_iou", avg_iou)
        self.log("valid/total_mAP", total_mAP)
        return {"valid/avg_val_iou": avg_iou, "log": logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
