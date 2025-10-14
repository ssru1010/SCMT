#!/usr/bin/env python
# ====================================================
# tests/test_interpretability.py — Explainability & visualization tests
# ====================================================

import torch
import numpy as np
from scmt.modeling_somt import SchemaAugmentedSOMT, SOMTConfig
from scmt.modeling_utils import visualize_schemas
from scmt.explain import generate_with_schema_attribution
from scmt.schema_analysis import compute_schema_similarity


class DummyTokenizer:
    def __init__(self): self.vocab = {i: str(i) for i in range(128)}
    def decode(self, ids, skip_special_tokens=True): return " ".join(str(i) for i in ids)
    def encode(self, text, add_special_tokens=False): return [int(x) for x in text.split() if x.isdigit()]
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

def _make_tiny_model():
    cfg = SOMTConfig(vocab_size=64, d_model=16, nhead=2, num_layers=1, mem_size=8)
    return SchemaAugmentedSOMT(**cfg.to_dict()), DummyTokenizer()

def test_visualize_schemas_runs(capsys):
    model, tok = _make_tiny_model()
    visualize_schemas(model, tok, top_k=3)
    out = capsys.readouterr().out
    assert "Schema" in out

def test_analyze_schemas_runs(tmp_path):
    model, tok = _make_tiny_model()
    from scmt.schema_analysis import analyze_schemas
    result = analyze_schemas(model, tok, top_k=3, save_path=tmp_path / "out.json")
    assert "positions" in result and "clusters" in result

def test_generate_with_schema_attribution_shape():
    model, tok = _make_tiny_model()
    result = generate_with_schema_attribution(model, tok, prompt="1 2 3", max_len=5, visualize_heatmap=False)
    assert "text" in result and isinstance(result["schema_traces"], np.ndarray)

def test_schema_similarity_matrix():
    model, _ = _make_tiny_model()
    sim, redund = compute_schema_similarity(model)
    assert sim.shape[0] == model.num_schemas
    assert isinstance(redund, list)
