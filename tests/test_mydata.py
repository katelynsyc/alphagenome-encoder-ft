from __future__ import annotations

import torch

from alphagenome_encoder_ft.mydata import JoresMPRADataset


def _make_rows(n: int = 8) -> list[dict[str, str]]:
    bases = "ACGT"
    return [
        {
            "sequence": (bases * 50)[: 170],
            "enrichment_cold": "1.0",
            "enrichment_dark": "2.0",
            "enrichment_light": "3.0",
            "enrichment_warm": "4.0",
            "enrichment_maize": "5.0",
        }
        for _ in range(n)
    ]


def _make_dataset(**overrides) -> JoresMPRADataset:
    kwargs = dict(
        rows=_make_rows(),
        use_adapters=False,
        sequence_length=170,
        reverse_complement=True,
        rc_prob=0.5,
        random_shift=True,
        shift_prob=0.5,
        max_shift=10,
        seed=42,
    )
    kwargs.update(overrides)
    return JoresMPRADataset(**kwargs)


def _draw_batch(dataset: JoresMPRADataset) -> torch.Tensor:
    return torch.stack([dataset[i][0] for i in range(len(dataset))])


def test_set_epoch_reproduces_uninterrupted_run_for_that_epoch():
    # Simulates an uninterrupted run: continue drawing from the same dataset object as
    # set_epoch is called for epochs 1, 2, 3 in sequence.
    uninterrupted = _make_dataset()
    for epoch in (1, 2, 3):
        uninterrupted.set_epoch(epoch)
        batch_at_epoch = _draw_batch(uninterrupted)

    # Simulates a resumed run: a brand-new process/dataset object that jumps straight to
    # epoch 3 (as if epochs 1-2 already happened in an earlier, now-dead process).
    resumed = _make_dataset()
    resumed.set_epoch(3)
    resumed_batch = _draw_batch(resumed)

    assert torch.equal(batch_at_epoch, resumed_batch)


def test_set_epoch_changes_the_augmentation_stream():
    dataset = _make_dataset()
    dataset.set_epoch(1)
    epoch1_batch = _draw_batch(dataset)

    dataset.set_epoch(2)
    epoch2_batch = _draw_batch(dataset)

    assert not torch.equal(epoch1_batch, epoch2_batch)


def test_set_epoch_is_a_no_op_when_reseed_per_epoch_disabled():
    dataset = _make_dataset(reseed_per_epoch=False)
    baseline_batch = _draw_batch(dataset)

    # With reseeding disabled, set_epoch must not touch the RNG -- the augmentation stream
    # just continues from wherever it already was, exactly like before this feature existed.
    dataset.set_epoch(5)
    continued_batch = _draw_batch(dataset)

    dataset2 = _make_dataset(reseed_per_epoch=False)
    dataset2.set_epoch(5)  # should be a no-op here too
    fresh_batch_after_noop_set_epoch = _draw_batch(dataset2)

    assert torch.equal(baseline_batch, fresh_batch_after_noop_set_epoch)
    assert not torch.equal(baseline_batch, continued_batch)  # stream still advanced normally
