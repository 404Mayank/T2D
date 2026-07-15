"""Unit checks for PCGrad / UW plumbing (no data). Run: python -m training.path_b.b1.test_balance_unit"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.path_b.b1.balance import (
    UncertaintyWeights,
    pcgrad_combine,
    pcgrad_project,
    shared_parameters,
)
from training.path_b.b1.model import AttnLSTM64


def test_project_conflict():
    g_i = torch.tensor([1.0, 0.0])
    g_j = torch.tensor([-1.0, 0.0])  # opposing
    out = pcgrad_project(g_i, g_j)
    # projection onto normal of g_j removes all of g_i along g_j → 0
    assert torch.allclose(out, torch.zeros(2), atol=1e-6), out


def test_project_aligned_unchanged():
    g_i = torch.tensor([1.0, 2.0])
    g_j = torch.tensor([0.5, 1.0])  # same direction
    out = pcgrad_project(g_i, g_j)
    assert torch.allclose(out, g_i), out


def test_combine_conflict_reduces_dot():
    g_a = torch.tensor([1.0, 0.0])
    g_b = torch.tensor([-1.0, 0.1])
    rng = torch.Generator().manual_seed(0)
    comb, stats = pcgrad_combine(g_a, g_b, rng=rng)
    assert stats["conflict"] == 1.0
    assert comb.numel() == 2


def test_combine_fully_opposing_zeros():
    """Paper-faithful |T|=2: both project to 0 when perfectly opposing."""
    g_a = torch.tensor([1.0, 0.0])
    g_b = torch.tensor([-1.0, 0.0])
    comb, stats = pcgrad_combine(g_a, g_b, rng=torch.Generator().manual_seed(0))
    assert stats["conflict"] == 1.0
    assert torch.allclose(comb, torch.zeros(2), atol=1e-6), comb


def test_combine_aligned_is_plain_sum():
    g_a = torch.tensor([1.0, 2.0])
    g_b = torch.tensor([0.5, 1.0])
    comb, stats = pcgrad_combine(g_a, g_b, rng=torch.Generator().manual_seed(1))
    assert stats["conflict"] == 0.0
    assert torch.allclose(comb, g_a + g_b), comb


def test_shared_params_exclude_attn():
    m = AttnLSTM64(d_in=4, hidden=8, n_classes=4, n_glu=8, bidirectional=True)
    shared = shared_parameters(m)
    shared_ids = {id(p) for p in shared}
    for p in m.attn.parameters():
        assert id(p) not in shared_ids
    for p in m.class_head.parameters():
        assert id(p) not in shared_ids
    for p in m.glu_head.parameters():
        assert id(p) not in shared_ids
    assert any(id(p) in shared_ids for p in m.input.parameters())
    assert any(id(p) in shared_ids for p in m.lstm.parameters())
    assert any(id(p) in shared_ids for p in m.proj.parameters())


def test_uw_init_matches_lambda05():
    uw = UncertaintyWeights()
    ce = torch.tensor(2.0)
    glu = torch.tensor(4.0)
    total, term_ce, term_glu = uw.combine(ce, glu, glu_active=True)
    # s=0 → exp(0)*2 + 0 + 0.5*exp(0)*4 + 0 = 2 + 2 = 4
    assert torch.allclose(total, torch.tensor(4.0)), total
    assert torch.allclose(term_ce, torch.tensor(2.0))
    assert torch.allclose(term_glu, torch.tensor(2.0))


def test_uw_zero_glu_still_has_reg():
    uw = UncertaintyWeights()
    ce = torch.tensor(1.0)
    glu = torch.tensor(99.0)
    total, term_ce, term_glu = uw.combine(ce, glu, glu_active=False)
    assert torch.allclose(total, torch.tensor(1.0)), total
    assert torch.allclose(term_glu, torch.tensor(0.0))


def test_pcgrad_backward_zero_glu_skip():
    from training.path_b.b1.balance import ConflictMeter, pcgrad_backward

    m = AttnLSTM64(d_in=4, hidden=8, n_classes=4, n_glu=2, bidirectional=True)
    x = torch.randn(2, 3, 4)
    wm = torch.ones(2, 3, dtype=torch.bool)
    out = m(x, wm)
    y = torch.tensor([0, 1])
    loss_ce = nn.functional.cross_entropy(out["logits"], y)
    loss_glu = out["glu_pred"].sum() * 0.0  # zero
    meter = ConflictMeter()
    stats = pcgrad_backward(
        model=m,
        loss_ce=loss_ce,
        loss_glu=loss_glu,
        glu_active=False,
        meter=meter,
    )
    assert stats["glu_active"] == 0.0
    assert meter.n_train_steps == 1
    assert meter.n_glu_active == 0
    shared = shared_parameters(m)
    assert any(p.grad is not None for p in shared)


def test_pcgrad_uw_s_grads_finite():
    """Integrated pcgrad_uw graph: s_* grads finite; model grads set."""
    from training.path_b.b1.balance import pcgrad_backward

    m = AttnLSTM64(d_in=4, hidden=8, n_classes=4, n_glu=2, bidirectional=True)
    uw = UncertaintyWeights()
    x = torch.randn(3, 4, 4)
    wm = torch.ones(3, 4, dtype=torch.bool)
    out = m(x, wm)
    y = torch.tensor([0, 1, 2])
    loss_ce = nn.functional.cross_entropy(out["logits"], y)
    gy = out["glu_pred"].detach() + 0.5
    gm = torch.ones(3, 4, dtype=torch.bool)
    mexp = gm.unsqueeze(-1).expand_as(out["glu_pred"])
    loss_glu = ((out["glu_pred"] - gy)[mexp]).pow(2).mean()
    total, term_ce, term_glu = uw.combine(loss_ce, loss_glu, glu_active=True)
    pcgrad_backward(
        model=m,
        loss_ce=term_ce,
        loss_glu=term_glu,
        glu_active=True,
        retain_graph=True,
    )
    s_grads = torch.autograd.grad(total, list(uw.parameters()), allow_unused=True)
    for g in s_grads:
        assert g is not None and torch.isfinite(g).all(), g
    assert m.proj.weight.grad is not None
    assert torch.isfinite(m.proj.weight.grad).all()


def main() -> int:
    tests = [
        test_project_conflict,
        test_project_aligned_unchanged,
        test_combine_conflict_reduces_dot,
        test_combine_fully_opposing_zeros,
        test_combine_aligned_is_plain_sum,
        test_shared_params_exclude_attn,
        test_uw_init_matches_lambda05,
        test_uw_zero_glu_still_has_reg,
        test_pcgrad_backward_zero_glu_skip,
        test_pcgrad_uw_s_grads_finite,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
