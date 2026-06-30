"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def correlation_metrics(preds: np.ndarray, targets: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute per-tissue Pearson correlation.

    Args:
        preds: predictions array of shape (N, num_tissues)
        targets: targets array of shape (N, num_tissues)
        prefix: key prefix, e.g. "val" or "test"

    Returns:
        dict mapping "{prefix}/{tissue}_pearson" to the Pearson r value.
    """
    out = {}
    for i, name in enumerate(["Leaf", "Fruit"]):
        out[f"{prefix}/{name}_pearson"] = float(pearsonr(preds[:, i], targets[:, i]).statistic)
    return out
