import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif hasattr(m, 'fc_self'):
        torch.nn.init.xavier_uniform_(m.fc_self.weight)
        if m.fc_self.bias is not None:
            m.fc_self.bias.data.fill_(0.01)
    elif hasattr(m, 'fc_neigh'):
        torch.nn.init.xavier_uniform_(m.fc_neigh.weight)
        if m.fc_neigh.bias is not None:
            m.fc_neigh.bias.data.fill_(0.01)

def compute_loss(pos_score, neg_score, weight_pos=1.0, weight_neg=1.0):
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score),
        weight=torch.full_like(pos_score, weight_pos)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_score, torch.zeros_like(neg_score),
        weight=torch.full_like(neg_score, weight_neg)
    )
    return pos_loss + neg_loss

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_score.shape[0]),
        torch.zeros(neg_score.shape[0])
    ]).detach().cpu().numpy()
    return roc_auc_score(labels, scores)

def calculate_rmse(pred_scores, true_weights):
    mse = np.mean((pred_scores.numpy() - true_weights.numpy()) ** 2)
    return math.sqrt(mse)
