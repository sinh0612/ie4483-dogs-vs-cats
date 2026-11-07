import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(counts, beta=0.999):
    # Cui et al. 2019: weight ~ 1 / effective number (1 - beta^n) / (1 - beta), normalised to sum to C
    counts = np.asarray(counts, dtype=np.float64)
    w = (1 - beta) / (1 - np.power(beta, counts))
    return w / w.sum() * len(counts)


def inverse_freq_weights(counts):
    w = 1.0 / np.asarray(counts, dtype=np.float64)
    return w / w.sum() * len(counts)


class LogitAdjustedLoss(nn.Module):
    # Menon et al. 2021: CE on logits + tau * log(prior); predict with the raw logits
    def __init__(self, counts, tau=1.0, label_smoothing=0.0):
        super().__init__()
        prior = torch.tensor(counts, dtype=torch.float32) / sum(counts)
        self.register_buffer("adjust", tau * torch.log(prior))
        self.label_smoothing = label_smoothing

    def forward(self, logits, y):
        return F.cross_entropy(logits + self.adjust, y, label_smoothing=self.label_smoothing)


class FocalLoss(nn.Module):
    # Lin et al. 2017: -(1 - p_t)^gamma * log p_t, no alpha term
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), 1).gather(1, y[:, None])[:, 0]
        return (-(1 - logp.exp()) ** self.gamma * logp).mean()


def build_criterion(method, counts, label_smoothing=0.0, beta=0.999, tau=1.0, gamma=2.0):
    """ce / resample: plain CE (resampling is done by the loader). cb, inv: weighted CE. la, focal."""
    if method in ("ce", "resample"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if method in ("cb", "inv"):
        w = class_balanced_weights(counts, beta) if method == "cb" else inverse_freq_weights(counts)
        return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=label_smoothing)
    if method == "la":
        return LogitAdjustedLoss(counts, tau, label_smoothing)
    if method == "focal":
        assert label_smoothing == 0, "focal loss is used without label smoothing"
        return FocalLoss(gamma)
    raise ValueError(method)
