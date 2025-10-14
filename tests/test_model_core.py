#!/usr/bin/env python
# ====================================================
# tests/test_model_core.py — Core model functionality
# ====================================================

import torch
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
