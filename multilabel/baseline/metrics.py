import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def All_metric(pred, gt, n_classes, type='train'):
    conf_mat=[]
    labels = [i for i in range(n_classes)]
    for label_col in range(len(labels)):
        y_true_label = pred[:, label_col]
        y_pred_label = gt[:, label_col]
        metric = []
        metric.append(accuracy_score(y_pred=y_pred_label, y_true=y_true_label))
        metric.append(recall_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0))
        metric.append(precision_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0))
        metric.append(f1_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0))
        try:
            metric.append(roc_auc_score(y_true_label, y_pred_label, multi_class = 'ovo')) # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        except:
            metric.append(roc_auc_score(y_true_label, y_pred_label))
        conf_mat.append(metric)
    if type=='train':
        return np.mean(np.array(conf_mat), axis=0)
    else:
        return np.array(conf_mat)
