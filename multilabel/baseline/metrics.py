import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def All_metric(pred, gt):
    assert pred.shape == gt.shape

    conf_mat=[]
    for label_col in range(gt.shape[1]):
        y_true_label = pred[:, label_col]
        y_pred_label = gt[:, label_col]
        metric = []
        # metric.append(accuracy_score(y_pred=y_pred_label, y_true=y_true_label)) #label acc
        metric.append(recall_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0))  #aR
        metric.append(precision_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0))   #aP
        metric.append(f1_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0))  #aF1
        metric.append(roc_auc_score(y_true_label, y_pred_label)) #aAUC

        conf_mat.append(metric)

    match_acc = np.mean((pred == gt).min(axis = 1))   #mean perfect match acc
    

    return np.mean(np.array(conf_mat), axis=1), match_acc

