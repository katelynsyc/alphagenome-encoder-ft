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
    for x, y, w in loader: #pulls one batch at a time from the DataLoader x = one-hot DNA (batch_size, 4 base, seq_length) ,y=target expression (4 samples, 2 outputs to pred), w=weights (4 sample weights)
        x, y, w = x.to(device), y.to(device), w.to(device)
        with torch.set_grad_enabled(train):
            yh = model(x) # (batch_size, num_outputs) this is predictions
            per_sample = loss_fn(yh, y).mean(dim=1) #for each sequence, it averages the loss across the diff conditions, so diff loss per each seq
            if train: #if training, it does backward pass and optimizer
                optim.zero_grad()
                loss = (per_sample * w).mean()
                loss.backward() #computes gradients with backpropagation
                optim.step() #updates parameters using computed gradients
            else:
                loss = per_sample.mean()
        total_loss += loss.item() * len(x) #accum total loss, as weighted by the batch size
        n_total += len(x)
        preds.append(yh.detach().cpu().numpy())
        targets.append(y.cpu().numpy())
    return total_loss / n_total, np.concatenate(preds), np.concatenate(targets)
