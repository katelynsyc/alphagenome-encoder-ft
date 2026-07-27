"""Regression tests for the tangermeme-facing ISM wrapper's axis handling.

AlphaGenomeEncoderModel expects channels-last input (N, L, C); tangermeme's
saturation_mutagenesis hard-codes channels-first (N, C, L). A stray or missing
transpose anywhere in that chain doesn't necessarily raise -- it can silently
produce a same-shaped, wrong result. These tests use a small synthetic model
(not the real checkpoint) so they run fast and isolate the axis-order logic
from model weights entirely.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyOrganismModel(nn.Module):
    """Stand-in for AlphaGenomeEncoderModel: forward(X, organism_idx) with
    channels-last X, shape (N, L, C). organism_idx is accepted but ignored,
    matching the real model's encoder_only path.

    Weights each position differently so the output actually depends on which
    axis is "length" -- a stray transpose changes the numbers, it doesn't just
    fail to run.
    """

    def __init__(self, length: int, channels: int):
        super().__init__()
        self.length = length
        self.channels = channels
        self.position_weight = nn.Parameter(torch.arange(length, dtype=torch.float32), requires_grad=False)

    def forward(self, X: torch.Tensor, organism_idx: torch.Tensor) -> torch.Tensor:
        if X.shape[1:] != (self.length, self.channels):
            raise ValueError(f"expected (N, {self.length}, {self.channels}), got {tuple(X.shape)}")
        weighted = X * self.position_weight.view(1, -1, 1)
        return weighted.sum(dim=1)  # (N, channels)


class TangermemeWrapper(nn.Module):
    """tangermeme builds X as (N, C, L); the real model wants (N, L, C)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, X: torch.Tensor, organism_idx: torch.Tensor) -> torch.Tensor:
        return self.model(X.transpose(1, 2), organism_idx)


def _make_onehot(n: int, length: int, channels: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, channels, (n, length), generator=generator)
    return F.one_hot(idx, num_classes=channels).float()  # (N, L, C)


def test_wrapper_matches_direct_model_call():
    """wrapped(X_in_tangermeme_layout, ...) must equal model(X_native_layout, ...)
    exactly -- this is the sanity check to run before trusting any ISM output."""
    model = DummyOrganismModel(length=6, channels=4)
    wrapped = TangermemeWrapper(model)

    X_native = _make_onehot(n=3, length=6, channels=4, seed=0)  # (N, L, C)
    organism_idx = torch.zeros(3, dtype=torch.long)

    direct = model(X_native, organism_idx)
    via_wrapper = wrapped(X_native.transpose(1, 2), organism_idx)  # (N, C, L), tangermeme's layout

    assert torch.equal(direct, via_wrapper)


def test_wrapper_wrong_layout_raises_when_shape_mismatches():
    """Forgetting the setup transpose is at least loud when length != channels."""
    model = DummyOrganismModel(length=6, channels=4)
    wrapped = TangermemeWrapper(model)

    X_native = _make_onehot(n=2, length=6, channels=4, seed=1)
    organism_idx = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError):
        wrapped(X_native, organism_idx)  # missing the (N,L,C) -> (N,C,L) setup transpose


def test_wrong_layout_is_silently_wrong_when_axes_happen_to_match():
    """The dangerous case: when length == channels, a missing/extra transpose
    produces a *different but same-shaped* result instead of an error -- the
    exact silent-corruption scenario the manual sanity check exists to catch."""
    length = channels = 4
    model = DummyOrganismModel(length=length, channels=channels)
    wrapped = TangermemeWrapper(model)

    X_native = _make_onehot(n=3, length=length, channels=channels, seed=2)
    organism_idx = torch.zeros(3, dtype=torch.long)

    correct = wrapped(X_native.transpose(1, 2), organism_idx)  # properly transposed into tangermeme's layout
    wrong = wrapped(X_native, organism_idx)  # setup transpose skipped by mistake

    assert correct.shape == wrong.shape  # no crash -- same shape either way
    assert not torch.allclose(correct, wrong)  # but numerically different: silently corrupted


def test_saturation_mutagenesis_y0_matches_wrapper_and_direct_model():
    """Full three-way check: saturation_mutagenesis's own y0[0] should equal both
    a direct wrapped() call and a direct model() call on the same reference
    sequence -- confirming the library's internal batching introduces no further
    reordering on top of the wrapper."""
    tangermeme_sm = pytest.importorskip("tangermeme.saturation_mutagenesis")

    length, channels = 6, 4
    model = DummyOrganismModel(length=length, channels=channels)
    wrapped = TangermemeWrapper(model)

    X_native = _make_onehot(n=3, length=length, channels=channels, seed=3)  # (N, L, C)
    X = X_native.transpose(1, 2)  # (N, C, L), what saturation_mutagenesis expects
    organism_idx = torch.zeros(X.shape[0], dtype=torch.long)

    y0, _y_hat = tangermeme_sm.saturation_mutagenesis(
        wrapped, X, args=(organism_idx,), raw_outputs=True,
    )

    check_a = model(X_native[:1], organism_idx[:1])
    check_b = wrapped(X[:1], organism_idx[:1])

    assert torch.allclose(check_a, check_b)
    assert torch.allclose(check_a, y0[:1])
