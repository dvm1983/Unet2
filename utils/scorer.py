import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr).view(-1)

    if threshold is not None:
        pr = (pr >= threshold).float()

    gt = (gt > 0).float().view(-1)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp    
    
    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    
    return score


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., 
                           eps=self.eps, threshold=None, 
                           activation=self.activation)


class Scorer:
    def __init__(self):
        self.scorers = {'dice(threshold=0.2)': lambda x, y: f_score(x, y, threshold=0.2),
                        'dice(threshold=0.4)': lambda x, y: f_score(x, y, threshold=0.4),
                        'dice(threshold=0.5)': lambda x, y: f_score(x, y, threshold=0.5),
                        'dice(threshold=0.6)': lambda x, y: f_score(x, y, threshold=0.6),
                        'dice(threshold=0.8)': lambda x, y: f_score(x, y, threshold=0.8),
                        'dice(threshold=0.9)': lambda x, y: f_score(x, y, threshold=0.9)}

    def __call__(self, pred, pred_proba, round=4):
        history = defaultdict(list)
        for scorer_name, scorer in self.scorers.items():
            history[scorer_name] = np.round(scorer(pred_proba, pred), round)
            
        return history