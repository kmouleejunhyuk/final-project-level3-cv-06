import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
import warnings


def All_metric(pred, gt, n_classes, score, type='train'):
    # pred, gt, score (batch_size, 38)
    # warnings.filterwarnings("ignore")
    # warnings.filterwarnings("error")

    conf_mat = []
    labels = [i for i in range(n_classes)]
    for label_col in range(len(labels)):
        y_true_label = pred[:, label_col]
        y_pred_label = gt[:, label_col]
        y_score = score[:, label_col]
        metric = []
        metric.append(accuracy_score(y_pred=y_pred_label, y_true=y_true_label))
        metric.append(recall_score(y_pred=y_pred_label,
                      y_true=y_true_label, zero_division=0))
        metric.append(precision_score(y_pred=y_pred_label,
                      y_true=y_true_label, zero_division=0))
        metric.append(f1_score(y_pred=y_pred_label,
                      y_true=y_true_label, zero_division=0))
        try:
            # fpr, tpr, _ = roc_curve(y_true_label, y_score)
            # roc_auc = auc(fpr, tpr)
            roc_auc = roc_auc_score(y_pred_label, y_score)
            metric.append(roc_auc)
        # print('roc_auc', roc_auc)

        except:
            # print('roc_auc', roc_auc)
            print(f'class:{label_col}')
            print('gt')
            print(np.count_nonzero(gt, axis=1))
            print('pred')
            print(np.count_nonzero(pred, axis=1))
            metric.append(0)


        conf_mat.append(metric)
    
        # warnings.filterwarnings("default")
    if type == 'train':
        return np.mean(np.array(conf_mat), axis=0)
    else:
        return np.array(conf_mat)
