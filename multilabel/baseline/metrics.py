import numpy as np
import torch
import sys
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
np.seterr(divide='ignore', invalid='ignore')

def All_metric(pred, gt):
    assert pred.shape == gt.shape
    conf_mat=[]
    for label_col in range(gt.shape[1]):
        y_true_label = pred[:, label_col]
        y_pred_label = gt[:, label_col]

        recall = recall_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0)
        precision = precision_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0)
        f1 = f1_score(y_pred=y_pred_label, y_true=y_true_label, zero_division=0)

        conf_mat.append([recall, precision, f1])

    EMR = np.mean((pred == gt).min(axis = 1))
    mean_mat = np.array(conf_mat)
    mean_mat = nonzero_mean(mean_mat, 0)
    mean_mat.append(EMR)
    return mean_mat, conf_mat
    
def nonzero_mean(x1, axis):
    mat = np.sum(x1, axis = axis) / np.count_nonzero(x1, axis = axis)
    return list(np.nan_to_num(mat, copy = True, nan = 0))
    
def top_k_labels(outs: torch.Tensor, k: torch.Tensor, identity = True):
    if identity:
        k = torch.argmax(k, dim = -1).clone().detach().tolist()
        inlet = outs.clone().detach()
        
        for row, n in zip(inlet, k):
            s = row.argsort(-1, descending=True)
            thr, rev = s[:n], s[n:]
            row[thr], row[rev] = 1, 0
    
        return inlet
    else:
        return torch.argmax(outs, dim = -1).clone().detach()

        
def get_confusion_matrix(pred, gt):
    _matrix = []
    
    for col in range(gt.shape[1]):
        tn, fp, fn, tp = 0,0,0,0
        g, p = gt[:, col], pred[:, col] 
        for g_, p_ in zip(g, p):
            if g_ and g_ == p_:
                tp +=1
            elif g_ and g_ != p_:
                fn +=1
            elif not g_ and g_ == p_:
                tn +=1
            elif not g_ and g_ != p_:
                fp +=1

        _matrix.append([tn, fp, fn, tp])

    return _matrix


def get_metrics_from_matrix(confusion_matrix):
    '''
    confusion matrix: tn, fp, fn, tp order
    '''
    epsilon = sys.float_info.epsilon
    label_metric = []
    for row in confusion_matrix:
        tn, fp, fn, tp = row
        recall = tp / (tp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        f1 = 2 / (1/(recall + epsilon) + 1/(precision + epsilon))
        label_metric.append([recall, precision, f1])

    label_metric = np.array(label_metric)
    mAR, mAP, mF1 = np.mean(label_metric, axis = 0)

    return label_metric, (mAR, mAP, mF1)


if __name__ == '__main__':
    ans = top_k_labels(torch.tensor([[2,1,3,4],[2,1,3,4]]), [2, 3])
    print(ans)