import torch
from effdet import (DetBenchTrain, EfficientDet, create_model,
                    get_efficientdet_config)
from effdet.efficientdet import HeadNet
from pytorch_lightning import LightningModule


def get_train_efficientdet():
    config = get_efficientdet_config('tf_efficientdet_d0')
    # net = create_model('tf_efficientdet_d5', bench_task='predict', num_classes=39, checkpoint_path='/opt/ml/finalproject/detection/models/weights/efficientdet_d5-ef44aea8.pth')
    config.num_classes = 39
    config.image_size = (1024, 1024)
    net = EfficientDet(config, pretrained_backbone=False)
    # checkpoint = torch.load('/opt/ml/finalproject/detection/models/weights/efficientdet_d5-ef44aea8.pth')
    # net.load_state_dict(checkpoint)
    
    net.class_net = HeadNet(config, num_outputs=config.num_classes) # , norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)

class WheatModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_train_efficientdet()
    
    def forward(self, image, target):
        return self.model(image, target)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images).float()
        targets = [{k: v for k, v in t.items()} for t in targets]

        targets2 = {}
        targets2["bbox"] = [
            target["boxes"].float() for target in targets
        ]  # variable number of instances, so the entire structure can be forced to tensor
        targets2["cls"] = [target["labels"].float() for target in targets]
        targets2["image_id"] = torch.tensor(
            [target["image_id"] for target in targets]
        ).float()
        targets2["img_scale"] = torch.tensor(
        [target["img_scale"] for target in targets], device="cuda"
        ).float()
        targets2["img_size"] = torch.tensor(
            [(1024, 1024) for target in targets]
        ).float()

        loss_dict = self.model(images, targets2)
        return {"loss": loss_dict["loss"], "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images).float()
        targets = [{k: v for k, v in t.items()} for t in targets]

        targets2 = {}
        targets2["bbox"] = [
            target["boxes"].float() for target in targets
        ]  # variable number of instances, so the entire structure can be forced to tensor
        targets2["cls"] = [target["labels"].float() for target in targets]
        targets2["image_id"] = torch.tensor(
            [target["image_id"] for target in targets]
        ).float()
        targets2["img_scale"] = torch.tensor(
        [target["img_scale"] for target in targets], device="cuda"
        ).float()
        targets2["img_size"] = torch.tensor(
            [(1024, 1024) for target in targets], device="cuda"
        ).float()

        loss_dict = self.model(images, targets2)
        loss_val = loss_dict["loss"]
        detections = loss_dict["detections"]
        # Back to xyxy format.
        detections[:, :, [1,0,3,2]] = detections[:, :, [0,1,2,3]]
        # xywh to xyxy => not necessary.
        # detections[:, :, 2] += detections[:, :, 0]
        # detections[:, :, 3] += detections[:, :, 1]

        res = {target["image_id"].item(): {
                    'boxes': output[:, 0:4],
                    'scores': output[:, 4],
                    'labels': output[:, 5]}
                for target, output in zip(targets, detections)}
        # iou = self._calculate_iou(targets, res, IMG_SIZE)
        # iou = torch.as_tensor(iou)
        # self.coco_evaluator.update(res)
        return {"loss": loss_val, "log": loss_dict}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-2)

