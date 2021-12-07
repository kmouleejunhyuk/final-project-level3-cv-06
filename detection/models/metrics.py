from collections import Counter

import torch
from torch import Tensor
from torchvision.ops import box_iou


# https://github.com/PyTorchLightning/pytorch-lightning/blob/64abecf04fd6882128fbd527bbfa394ea981dc5b/pytorch_lightning/metrics/functional/object_detection.py
def mean_average_precision(
    pred_image_indices: Tensor,
    pred_probs: Tensor,
    pred_labels: Tensor,
    pred_bboxes: Tensor,
    target_image_indices: Tensor,
    target_labels: Tensor,
    target_bboxes: Tensor,
    iou_threshold: float
) -> Tensor:
    # pred_label = [0,3,7,19] target_label = [1,3,8,19,20] -> [0,3,7,19,1,3,8,19,20] -> [0,1,3,7,8,19,20]
    # {0:0.5, 1:0.4 ..., 20:0.9}
    classes = torch.cat([pred_labels, target_labels]).unique()
    average_precisions = torch.zeros(len(classes))
    class_ap_dict = {str(i):-1. for i in range(39)}
    for class_idx, c in enumerate(classes):
        # Descending indices w.r.t. class probability for class c
        desc_indices = torch.argsort(pred_probs, descending=True)[pred_labels == c]
        # No predictions for this class so average precision is 0
        if len(desc_indices) == 0:
            continue
        targets_per_images = Counter([idx.item() for idx in target_image_indices[target_labels == c]])
        targets_assigned = {
            image_idx: torch.zeros(count, dtype=torch.bool) for image_idx, count in targets_per_images.items()
        }
        tps = torch.zeros(len(desc_indices))
        fps = torch.zeros(len(desc_indices))
        for i, pred_idx in enumerate(desc_indices):
            image_idx = pred_image_indices[pred_idx].item()
            # Get the ground truth bboxes of class c and the same image index as the prediction
            gt_bboxes = target_bboxes[(target_image_indices == image_idx) & (target_labels == c)]
            ious = box_iou(torch.unsqueeze(pred_bboxes[pred_idx], dim=0), gt_bboxes)
            best_iou, best_target_idx = ious.squeeze(0).max(0) if len(gt_bboxes) > 0 else (0, -1)
            # Prediction is a true positive is the IoU score is greater than the threshold and the
            # corresponding ground truth has only one prediction assigned to it
            if best_iou > iou_threshold and not targets_assigned[image_idx][best_target_idx]:
                targets_assigned[image_idx][best_target_idx] = True
                tps[i] = 1
            else:
                fps[i] = 1
        tps_cum, fps_cum = torch.cumsum(tps, dim=0), torch.cumsum(fps, dim=0)
        precision = tps_cum / (tps_cum + fps_cum)
        num_targets = len(target_labels[target_labels == c])
        recall = tps_cum / num_targets if num_targets else tps_cum
        precision = torch.cat([reversed(precision), torch.tensor([1.])])
        recall = torch.cat([reversed(recall), torch.tensor([0.])])

        average_precision = 0
        recall_thresholds = torch.linspace(0, 1, 101)
        for threshold in recall_thresholds:
            points = precision[:-1][recall[:-1] >= threshold]
            average_precision += torch.max(points) / 101 if len(points) else 0
        
        class_ap_dict[str(c.item())] = float(average_precision)

        average_precisions[class_idx] = average_precision
    mean_average_precision = torch.mean(average_precisions)
    return mean_average_precision, class_ap_dict
