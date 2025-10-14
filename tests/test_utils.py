#!/usr/bin/env python
# ====================================================
# tests/test_utils.py — Utility and helper checks
# ====================================================

import torch
from scmt.modeling_utils import top_k_top_p_filtering
from scmt.metrics import perplexity_from_loss, schema_usage_stats


def test_top_k_top_p_filtering_stability():
    logits = torch.randn(2, 20)
    filtered = top_k_top_p_filtering(logits.clone(), top_k=5, top_p=0.8)
    assert torch.isfinite(filtered).all()
    assert filtered.shape == logits.shape

def test_perplexity_conversion():
    loss = 1.5
    ppl = perplexity_from_loss(loss)
    assert abs(ppl - torch.exp(torch.tensor(loss))) < 1e-6

def test_schema_usage_stats_output():
    import numpy as np
    util = np.arange(5)
    counts = np.arange(5) + 1
    result = schema_usage_stats(util, util, counts)
    assert "utility" in result and "avg_util" in result
