import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
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
    mean_mat = nonzero_mean(mean_mat,gt, 0)
    mean_mat.append(EMR)
    return mean_mat, conf_mat
    
def nonzero_mean(x1, ref, axis):
    nonzero = np.count_nonzero(np.count_nonzero(ref, axis = 1), axis = 0)   #38 -> 1
    mat = np.sum(x1, axis = axis) / nonzero
    return list(np.nan_to_num(mat, copy = True, nan = 0))
    
def top_k_labels(outs: torch.Tensor, k: torch.Tensor):
    k = torch.argmax(k, dim = -1).clone().detach().tolist()
    inlet = outs.clone().detach()
    
    for row, n in zip(inlet, k):
        s = row.argsort(-1, descending=True)
        thr, rev = s[:n], s[n:]
        row[thr], row[rev] = 1, 0
    
    return inlet

if __name__ == '__main__':
    ans = top_k_labels(torch.tensor([[2,1,3,4],[2,1,3,4]]), [2, 3])
    print(ans)