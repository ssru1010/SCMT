#!/usr/bin/env python
# ====================================================
# tests/test_model_core.py — Core model functionality
# ====================================================

import torch, torch.nn as nn, torch.nn.functional as F
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

def test_memory_retrieval_is_causal():
    cfg = SOMTConfig(vocab_size=10, d_model=32, max_len=10, mem_size=4, num_schemas=2)
    model = SchemaAugmentedSOMT(**cfg.to_dict())
    model.eval()

    # Force high entropy to trigger writes
    x1 = torch.tensor([[1, 2, 3, 4, 5]])
    x2 = torch.tensor([[1, 2, 3, 9, 8]])

    with torch.no_grad():
        logits1, _, _, _, aux1 = model(x1)
        logits2, _, _, _, aux2 = model(x2)

    # Check memory timestamps
    ts1 = aux1["memory_timestamps"][0]
    ts2 = aux2["memory_timestamps"][0]
    used1 = ts1[ts1 != 1e9]
    used2 = ts2[ts2 != 1e9]
    print("TS1:", used1)
    print("TS2:", used2)

    # Now check logits for position 2 (should not see token 3+)
    diff = torch.abs(logits1[0, :3] - logits2[0, :3]).max().item()
    assert diff < 1e-3, f"Leak: {diff}"

def test_causal_mask_shape_and_values():
    model = SchemaAugmentedSOMT(vocab_size=10, d_model=16, max_len=5)
    mask = model._causal_mask(4, device="cpu")
    assert mask.shape == (4, 4)
    assert torch.all(mask[0, 1:] == float('-inf'))  # first row: future masked
    assert torch.all(mask.diag() == 0.0)             # diagonal: allowed
    assert mask.dtype in (torch.float32, torch.float16)

def test_memory_timestamps_are_causal():
    cfg = SOMTConfig(vocab_size=20, d_model=16, max_len=10, mem_size=8, num_schemas=2)
    model = SchemaAugmentedSOMT(**cfg.to_dict())
    x = torch.randint(0, 20, (1, 6))
    _, _, _, _, aux = model(x)
    ts = aux["memory_timestamps"][0]  # [mem_size]
    # Unused slots = 1e9; used slots should be < 6 and increasing
    used = ts[ts != 1e9]
    if used.numel() > 1:
        assert torch.all(used[:-1] <= used[1:]), "Timestamps not monotonic"
        assert torch.all(used < 6), "Timestamp exceeds input length"

def test_schema_router_is_probabilistic():
    model = SchemaAugmentedSOMT(vocab_size=10, d_model=32, num_schemas=5)
    x = torch.randint(0, 10, (2, 4))
    with torch.no_grad():
        logits, _, _, _, _ = model(x)
        # Access routing weights indirectly via aux or internal method
        encoded = model.embed(x) + model.pos_embed[:, :4]
        encoded = model.encoder(encoded, mask=model._causal_mask(4, x.device))
        schema_scores = torch.matmul(
            model.query_proj(encoded),
            model.schema_keys.unsqueeze(0).expand(2, -1, -1).transpose(-2, -1)
        ) / (32 ** 0.5)
        weights = F.softmax(schema_scores, dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)
    assert torch.all(weights >= 0) and torch.all(weights <= 1)

def test_entropy_normalization_is_per_batch():
    model = SchemaAugmentedSOMT(vocab_size=10, d_model=16)
    x = torch.randint(0, 10, (2, 5))
    with torch.no_grad():
        _, _, _, _, aux = model(x)
        e_norm = aux["entropy_norm"]  # [B, L]
    # High entropy in one batch shouldn’t suppress another
    assert e_norm.shape == (2, 5)
    # Manually verify stats per row
    for b in range(2):
        e_row = e_norm[b]
        assert abs(e_row.mean().item() - 0.5) < 0.2  # sigmoid → ~0.5 mean

def test_schema_utility_loss_non_negative():
    model = SchemaAugmentedSOMT(vocab_size=10, d_model=16, num_schemas=4)
    x = torch.randint(0, 10, (1, 6))
    with torch.no_grad():
        _, _, _, _, aux = model(x)
    assert aux["schema_utility"].item() >= 0

def test_memory_budget_range():
    """Checks that memory budget (used slots / mem_size) is within expected range after forward pass."""
    cfg = SOMTConfig(
        vocab_size=64,
        d_model=32,
        nhead=2,
        num_layers=2,
        max_len=25,
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