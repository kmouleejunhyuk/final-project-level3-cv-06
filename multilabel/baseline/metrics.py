import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc



def All_metric(pred, gt, n_classes, type='train'):
    # pred, gt, score (batch_size, 38) array

    # conf_mat shape : 38, 5
    conf_mat = []
    labels = [i for i in range(n_classes)]
    for label_col in range(len(labels)):
        y_true_label = pred[:, label_col]
        y_pred_label = gt[:, label_col]
        metric = []
        metric.append(accuracy_score(y_pred=y_pred_label, y_true=y_true_label))
        metric.append(recall_score(y_pred=y_pred_label,
                      y_true=y_true_label, zero_division=0))
        metric.append(precision_score(y_pred=y_pred_label,
                      y_true=y_true_label, zero_division=0))
        metric.append(f1_score(y_pred=y_pred_label,
                      y_true=y_true_label, zero_division=0))
        conf_mat.append(metric)
    
        # warnings.filterwarnings("default")
    if type == 'train':
        return np.mean(np.array(conf_mat), axis=0) , modified_acc(pred, gt)
    else:
        return np.array(conf_mat), modified_acc(pred, gt)


def modified_acc(pred, gt):
    score = 0
    for sample_pred, sample_gt in zip(pred,gt):
        if np.array_equal(sample_pred, sample_gt):
            score += 1
    return score/len(pred)
