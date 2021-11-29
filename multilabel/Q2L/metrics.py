import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def accuracy(pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
    return torch.mean((pred == ground_truth).to(float)).item()

def calculate_metrics(pred, target):
    pred, target = pred.detach().cpu().numpy(), target.detach().cpu().numpy()

    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division = 1),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division = 1),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division = 1),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division = 1),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division = 1),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division = 1),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples', zero_division = 1),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples', zero_division = 1),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples', zero_division = 1),
            }

def overall(pred: torch.Tensor, ground_truth: torch.Tensor , label_count: int = 39) -> float:
    # pred, ground truth must be hard
    optionlist = ['OR', 'OP', 'CP', 'CR', 'OF1', 'CF1']
    assert pred.shape == ground_truth.shape

    Mc, Mp, Mg = [], [], []
    for i in range(pred.shape[-1]):
        Mc_i, Mp_i, Mg_i = 0, 0, 0
        for j in range(pred.shape[0]):
            g, p = int(ground_truth[j, i]), int(pred[j, i])
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
    CP = sum([x / y if y else 0 for x, y in zip(Mc, Mp)]) / pred.shape[-1]
    CR = sum([x / y if y else 0 for x, y in zip(Mc, Mg)]) / pred.shape[-1]
    OF1 = 2*OP*OR/(OP+OR) 
    CF1 = 2*CP*CR/(CP+CR) if (CP + CR) else 0

    optiondict = [(name, x) for name, x in zip(optionlist, [OR, OP, CP, CR, OF1, CF1])]
    return optiondict


def AUC(pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
    pred, ground_truth = pred.detach().cpu().numpy(), ground_truth.detach().cpu().numpy()
    return metrics.roc_auc_score(ground_truth, pred, multi_class = 'ovo')
    

def dicts2dictlist(data: list):
    out = {[(key, []) for key in data[0].keys()]}
    for d in data:
        for key, val in d.items():
            out[key].append(val)
    
    return out


def get_boxplot(data: list):
    data = dicts2dictlist(data)
    optionlist = ['OR', 'OP', 'CP', 'CR', 'OF1', 'CF1']
    # plt 기본 스타일 설정
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots()
    ax.boxplot([data[k] for k in optionlist], notch=True)
    plt.xticks(
        [1, 2, 3, 4, 5, 6], 
        optionlist
    )

    return fig