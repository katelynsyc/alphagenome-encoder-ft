"""Training loop primitives for the Deng/DeepTOMATO model."""

from __future__ import annotations

import numpy as np
import torch


def run_epoch(model, loader, loss_fn, optim, device, train: bool):
    """Run one training or evaluation epoch.

    Returns (mean_loss, predictions_array, targets_array).
    """
    model.train(train)
    total_loss, n_total = 0.0, 0
    preds, targets = [], []
    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        with torch.set_grad_enabled(train):
            yh = model(x)
            per_sample = loss_fn(yh, y).mean(dim=1)
            if train:
                optim.zero_grad()
                loss = (per_sample * w).mean()
                loss.backward()
                optim.step()
            else:
                loss = per_sample.mean()
        total_loss += loss.item() * len(x)
        n_total += len(x)
        preds.append(yh.detach().cpu().numpy())
        targets.append(y.cpu().numpy())
    return total_loss / n_total, np.concatenate(preds), np.concatenate(targets)
