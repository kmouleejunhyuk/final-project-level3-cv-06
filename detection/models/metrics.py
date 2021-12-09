from collections import Counter

import numpy as np
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


class ConfusionMatrix:
    def __init__(self, num_classes=39, conf_threshold=0.3, iou_threshold=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def process_batch(self,
                      pred_probs: Tensor,
                      pred_labels: Tensor,
                      pred_bboxes: Tensor,
                      target_labels: Tensor,
                      target_bboxes: Tensor
    ):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = target_labels.astype(np.int16)

        try:
            detections = pred_probs[pred_probs > self.conf_threshold]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(target_labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return
        
        detection_classes = pred_labels.astype(np.int16)

        all_ious = box_iou(torch.unsqueeze(pred_bboxes, dim=0), target_bboxes)
        want_idx = np.where(all_ious > self.iou_threshold)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                        for i in range(want_idx[0].shape[0])]
        
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0: # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(target_labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1
        
        for i, detection in enumerate(pred_labels):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1
    
    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))
