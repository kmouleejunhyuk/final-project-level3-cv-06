import torch
import numpy as np
from sklearn import metrics

def accuracy(pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
    return torch.mean((pred == ground_truth).to(float)).item()


def overall(pred: torch.Tensor, ground_truth: torch.Tensor, option , label_count: int = 39):
    assert pred.shape == ground_truth.shape
    assert option in ['recall', 'precision']

    pf, gf = pred.flatten(), ground_truth.flatten()

    confusion_matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for i, (predict, ground) in enumerate(zip(pf, gf)):
        if ground:
            if predict == ground:
                confusion_matrix['tp'] += 1
            else:
                confusion_matrix['fn'] += 1
        else:
            if predict == ground:
                confusion_matrix['tn'] += 1
            else:
                confusion_matrix['fp'] += 1

    if option == 'recall':
        return confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'])
        
    if option == 'precision':
        return confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp'])



def AUC(pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
    pred, ground_truth = pred.numpy(), ground_truth.numpy()
    return metrics.roc_auc_score(ground_truth, pred, multi_class = 'ovo')
    