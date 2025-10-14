"""Utility helpers for SOMT package.

Contains small helpers for config I/O and sampling utilities.
"""
import json
from typing import Any, Dict
import os
import torch
import torch.nn.functional as F


def _save_config(config: Dict[str, Any], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _load_config(load_dir: str) -> Dict[str, Any]:
    path = os.path.join(load_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.json not found in {load_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Nucleus + top-k filtering for logits (batch-first).
    logits: (batch, vocab)
    Returns filtered logits with -inf where removed.
    """
    logits = logits.clone()
    batch_size = logits.size(0)
    vocab_size = logits.size(-1)

    if top_k > 0:
        top_k = min(max(top_k, 1), vocab_size)
        kth_vals, _ = torch.topk(logits, top_k, dim=-1)
        min_vals = kth_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_vals, torch.full_like(logits, -float("inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        # keep at least one
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        for b in range(batch_size):
            remove_idx = sorted_indices[b, sorted_indices_to_remove[b]]
            logits[b, remove_idx] = -float("inf")

    return logits

# ====================================================
# Interpretability & Visualization Utilities
# ====================================================

@torch.no_grad()
def visualize_schemas(model, tokenizer, top_k: int = 8):
    """
    Visualize semantic correspondence between learned schema keys
    and token representations in the model's key-projection space.

    Args:
        model: SchemaAugmentedSOMT instance (must have .embed, .key_proj, .schema_keys)
        tokenizer: Tokenizer supporting .decode()
        top_k: number of top tokens to display for each schema key
    """
    model.eval()
    device = next(model.parameters()).device

    # Raw embedding matrix
    embed_matrix = model.embed.weight.data  # (V, D)
    projected_tokens = model.key_proj(embed_matrix)  # (V, D)

    # Normalize for cosine similarity
    schema_norm = F.normalize(model.schema_keys, p=2, dim=-1)  # (S, D)
    token_norm = F.normalize(projected_tokens, p=2, dim=-1)    # (V, D)

    # Cosine similarity between schema and token vectors
    sim = torch.mm(schema_norm, token_norm.T)  # (S, V)

    print(f"\n=== Schema Interpretability (top {top_k} tokens per schema) ===\n")
    for i in range(sim.size(0)):
        top_vals, top_ids = torch.topk(sim[i], k=top_k, largest=True)
        tokens = [tokenizer.decode([tid.item()]).strip() for tid in top_ids]
        print(f"Schema {i:02d}: " + " | ".join(f"{tok} ({val:.3f})" for tok, val in zip(tokens, top_vals)))

    print("\nDone.\n")
