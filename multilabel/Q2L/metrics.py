import torch
import numpy as np
from sklearn import metrics

def accuracy(pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
    return torch.mean((pred == ground_truth).to(float)).item()


def overall(pred: torch.Tensor, ground_truth: torch.Tensor, option , label_count: int = 39, thr = 0.5) -> float:
    # pred must be soft labels, ground truth must be hard
    optionlist = ['OR', 'OP', 'CP', 'CR', 'OF1', 'CF1']
    assert pred.shape == ground_truth.shape
    assert option in optionlist
    img_count = pred.shape[0]

    Mc, Mp, Mg = [], [], []
    for i in range(pred.shape[-1]):
        Mc_i, Mp_i, Mg_i = 0, 0, 0
        for j in range(pred.shape[0]):
            g, p = int(ground_truth[j, i]), int(pred[j, i] > thr)
            Mg_i += 1
            if g==p:
                Mc_i += 1
            elif p:
                Mp_i += 1


        Mc.append(Mc_i)
        Mp.append(Mp_i)
        Mg.append(Mg_i)
            
    Mp = [x if x else y for x, y in zip(Mp, Mc)]
    OP = sum(Mc) / sum(Mp)
    OR = sum(Mc) / sum(Mg)
    CP = sum([x / y for x, y in zip(Mc, Mp)]) / pred.shape[-1]
    CR = sum([x / y for x, y in zip(Mc, Mg)]) / pred.shape[-1]
    OF1 = 2*OP*OR/(OP+OR)
    CF1 = 2*CP*CR/(CP+CR)

    optiondict = [(name, x) for name, x in zip(optionlist, [OR, OP, CP, CR, OF1, CF1])]
    return optiondict[option]


def AUC(pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
    pred, ground_truth = pred.numpy(), ground_truth.numpy()
    return metrics.roc_auc_score(ground_truth, pred, multi_class = 'ovo')
    