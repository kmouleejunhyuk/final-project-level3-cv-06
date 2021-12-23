from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MAP
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou

from models.metrics import ConfusionMatrix
from models.save_fig import inference_figure

class LitModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = self.config.num_classes # include background (0: background)

        classes_dict = {}
        for i in range(self.num_classes-1):
            classes_dict[i+1] = self.config.classes[i]

        self.classes = classes_dict
        self.model = fasterrcnn_resnet50_fpn(num_classes=self.num_classes)
        
        self.val_map = MAP(class_metrics=True, dist_sync_on_step=True, )
        self.conf_mat = ConfusionMatrix(num_classes=self.num_classes-1, 
                                        CONF_THRESHOLD=self.config.conf_thres, 
                                        IOU_THRESHOLD=self.config.iou_thres)
        self.conf_mat_columns = [v for k, v in self.classes.items()]
        self.cnt = 0

    def forward(self, imgs): # 필요 여부 논의
        self.model.eval()
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
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

    def on_validation_epoch_start(self) -> None:
        self.val_map.reset()
        self.cnt = 0

    def validation_step(self, batch, batch_idx):
        self.cnt += 1
        imgs, targets = batch # imgs = [batch, 3, img_size, img_size] / targets = [{'boxes':[], 'labels':[]}]
        outs = self.model(imgs) # outs = [{'boxes':[], 'labels':[], 'scores':[]}]
        preds, target = [], []
        for j, o in enumerate(outs):
            pred_boxes = o['boxes']
            pred_labels = o['labels']
            pred_scores = o['scores']
            
            target_boxes = targets[j]['boxes']
            target_labels = targets[j]['labels']
            
            preds.append({
                'boxes': pred_boxes,
                'labels': pred_labels,
                'scores': pred_scores
            })        
            target.append({
                'boxes': target_boxes,
                'labels': target_labels
            })

            pred = np.array([[*box.cpu().numpy(), score.cpu().item(), label.cpu().item()] for box, score, label in zip(pred_boxes, pred_scores, pred_labels)])
            label = np.array([[label.cpu().item(), *box.cpu().numpy()] for label, box in zip(target_labels, target_boxes)])

        if self.cnt % 170 == 0:
            figure = inference_figure(imgs, preds, target, self.classes, save_dir=self.config.save_dir)
            figure.savefig(Path(self.config.save_dir) / 'valid_inference.png', dpi=250)
            self.logger.log_image(key="inference", images=[self.config.save_dir+'/valid_inference.png'])

        self.val_map.update(preds=preds, target=target)
        self.conf_mat.process_batch(detections=pred, labels=label)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.global_step != 0:
            print(
                f"Running val metric on {len(self.val_map.groundtruth_boxes)} samples"
            )
            result = self.val_map.compute()
            self.log("valid/val_mAP", result['map'])
            self.log("valid/val_mAP_50", result['map_50'])
            self.log("valid/val_mAP_75", result['map_75'])
            self.log("valid/val_mAP_s", result['map_small'])
            self.log("valid/val_mAP_m", result['map_medium'])
            self.log("valid/val_mAP_l", result['map_large'])
            for i,v in enumerate(result['map_per_class'].tolist()):
                self.log(f"classes/{self.classes[int(i)+1]}", v)
        
        self.conf_mat.plot(save_dir=self.config.save_dir, names=self.conf_mat_columns)
        self.logger.log_image(key="confusion", images=[self.config.save_dir+'/confusion_matrix.png'])

    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        else:
            raise Exception("optimizer does not exist")
        return optimizer
