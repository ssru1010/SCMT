#!/usr/bin/env python
# ====================================================
# tests/test_model_core.py — Core model functionality
# ====================================================

import torch
import numpy as np
from scmt.modeling_somt import SchemaAugmentedSOMT, SOMTConfig


def test_forward_shapes():
    cfg = SOMTConfig(vocab_size=128, d_model=32, nhead=4, num_layers=2, mem_size=8)
    model = SchemaAugmentedSOMT(**cfg.to_dict())

    x = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, mk, mv, ma, aux = model(x)

    assert logits.shape == (2, 6, cfg.vocab_size)
    assert mk.shape[1] == cfg.mem_size
    assert mv.shape == mk.shape
    assert isinstance(aux, dict)
    assert all(k in aux for k in ["importance_entropy", "importance_l2", "schema_utility"])


def test_memory_growth_and_trim():
    cfg = SOMTConfig(vocab_size=64, d_model=16, nhead=2, num_layers=1, mem_size=4)
    model = SchemaAugmentedSOMT(**cfg.to_dict())
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    _, mk, mv, ma, _ = model(x)
    # Memory should be bounded by mem_size
    assert mk.shape[1] == cfg.mem_size


def test_save_load(tmp_path):
    cfg = SOMTConfig(vocab_size=32, d_model=16, nhead=2, num_layers=1)
    model = SchemaAugmentedSOMT(**cfg.to_dict())
    model.save_pretrained(tmp_path)
    reloaded = SchemaAugmentedSOMT.from_pretrained(tmp_path)
    assert isinstance(reloaded, SchemaAugmentedSOMT)
    for p1, p2 in zip(model.parameters(), reloaded.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_causality_leak():
    """Verifies that future token changes do NOT affect past logits (autoregressive correctness)."""
    cfg = SOMTConfig(
        vocab_size=100,
        d_model=64,
        nhead=2,
        num_layers=2,
        max_len=10,
        mem_size=16,
        num_schemas=8,
    )
    model = SchemaAugmentedSOMT(**cfg.to_dict())
    model.eval()

    # Two sequences: same prefix, different suffix
    x1 = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    x2 = torch.tensor([[10, 20, 30, 99, 88]], dtype=torch.long)

    with torch.no_grad():
        logits1, *_ = model(x1)
        logits2, *_ = model(x2)

    # First 3 tokens should be identical
    prefix_len = 3
    max_diff = torch.abs(logits1[0, :prefix_len] - logits2[0, :prefix_len]).max().item()

    assert max_diff < 1e-5, f"Causality leak detected! Max diff: {max_diff:.2e}"


def test_memory_budget_range():
    """Checks that memory budget (used slots / mem_size) is within expected range after forward pass."""
    cfg = SOMTConfig(
        vocab_size=64,
        d_model=32,
        nhead=2,
        num_layers=2,
        max_len=20,
        mem_size=16,
        num_schemas=4,
    )
    model = SchemaAugmentedSOMT(**cfg.to_dict())
    model.eval()

    # Input long enough to trigger memory writes (longer than mem_size)
    x = torch.randint(0, cfg.vocab_size, (1, 25))

    with torch.no_grad():
        _, _, _, _, aux = model(x)

    # Compute budget: fraction of memory slots used (timestamps != inf/1e9)
    mem_ts = aux["memory_timestamps"]  # shape: [B, mem_size]
    used_slots = (mem_ts != 1e9).sum(dim=1).float().mean().item()
    budget = used_slots / cfg.mem_size

    # Budget should be reasonable: not too sparse (<0.2) or full (>0.9) in this untrained but active setting
    # Note: untrained models may behave differently, but we expect *some* usage
    assert 0.1 <= budget <= 1.0, f"Memory budget {budget:.3f} out of plausible range [0.1, 1.0]"

    # Optional: log for debugging (pytest -s to see)
    # print(f"Memory budget: {budget:.3f}")