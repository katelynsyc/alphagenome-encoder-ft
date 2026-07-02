"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

#condition names per output width -- combine_fruit=True yields 2 outputs, False yields 4 (see mydata.py)
CONDITION_NAMES = {
    2: ["Leaf", "Fruit"],
    4: ["Leaf", "MG", "Br", "RR"],
}


def correlation_metrics(preds: np.ndarray, targets: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute per-condition Pearson correlation.

    Args:
        preds: predictions array of shape (N, num_outputs)
        targets: targets array of shape (N, num_outputs)
        prefix: key prefix, e.g. "val" or "test"

    Returns:
        dict mapping "{prefix}/{condition}_pearson" to the Pearson r value.
    """
    num_outputs = preds.shape[1]
    if num_outputs not in CONDITION_NAMES:
        raise ValueError(
            f"No condition names registered for {num_outputs} outputs; expected one of {sorted(CONDITION_NAMES)}"
        )
    names = CONDITION_NAMES[num_outputs]
    out = {}
    for i, name in enumerate(names):
        out[f"{prefix}/{name}_pearson"] = float(pearsonr(preds[:, i], targets[:, i]).statistic)
    return out
