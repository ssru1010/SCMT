#!/usr/bin/env python
# ====================================================
# explain.py — Explainable Generation with Schema Attribution
# ====================================================
# Dependencies: torch, matplotlib, seaborn (optional for heatmaps)
# Usage:
#   python explain.py --model ./checkpoints/somt --prompt "Once upon a time"
# ====================================================

import os, argparse, json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from modeling_somt import SchemaAugmentedSOMT
from modeling_utils import top_k_top_p_filtering


@torch.no_grad()
def generate_with_schema_attribution(
    model,
    tokenizer,
    prompt: str,
    max_len: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    visualize_heatmap: bool = True,
):
    """
    Generate text while capturing per-step schema activations.
    Returns the generated text, tokens, and schema activation map.
    """
    model.eval()
    device = next(model.parameters()).device

    toks = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([toks], dtype=torch.long, device=device)

    generated, mem_k, mem_v, mem_age = input_ids[0].tolist(), None, None, None
    schema_traces = []

    for step in range(max_len):
        logits, mem_k, mem_v, mem_age, _ = model(input_ids, mem_k, mem_v, mem_age)
        next_logits = logits[:, -1, :]

        # Repetition penalty
        for token_id in set(generated):
            next_logits[:, token_id] /= repetition_penalty

        next_logits = next_logits / max(temperature, 1e-6)
        next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = int(next_token.item())
        generated.append(token_id)

        # Schema activation snapshot
        if hasattr(model, "schema_keys"):
            with torch.no_grad():
                encoded = model.embed(torch.tensor([[token_id]], device=device))
                query = model.query_proj(encoded)
                schema_scores = torch.matmul(
                    query, model.schema_keys.T
                ) / np.sqrt(model.d_model)
                schema_probs = F.softmax(schema_scores, dim=-1).squeeze().cpu().numpy()
                schema_traces.append(schema_probs)

        if token_id in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
            break

        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    schema_traces = np.array(schema_traces)  # (T, num_schemas)

    if visualize_heatmap and schema_traces.shape[0] > 1:
        _plot_schema_heatmap(schema_traces, tokenizer, generated, model.num_schemas)

    return {"text": text, "schema_traces": schema_traces, "tokens": generated}


def _plot_schema_heatmap(schema_traces, tokenizer, tokens, num_schemas):
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.heatmap(schema_traces.T, cmap="viridis", cbar=True)
    plt.yticks(range(num_schemas), [f"S{i}" for i in range(num_schemas)], fontsize=6)
    token_labels = [tokenizer.decode([t]) for t in tokens]
    plt.xticks(range(len(tokens)), token_labels, rotation=90, fontsize=6)
    plt.title("Schema Activation Heatmap")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def schema_ablation_test(model, tokenizer, prompt: str, max_len: int = 60):
    """
    Test schema ablation: disables one schema at a time
    and measures perplexity degradation.
    """
    base_output = generate_with_schema_attribution(model, tokenizer, prompt, max_len=max_len, visualize_heatmap=False)
    base_ppl = _estimate_perplexity(model, tokenizer, prompt)

    ablation_scores = {}
    for i in range(model.num_schemas):
        backup_keys = model.schema_keys.clone()
        model.schema_keys[i] = 0.0
        ablated_ppl = _estimate_perplexity(model, tokenizer, prompt)
        ablation_scores[f"schema_{i}"] = ablated_ppl - base_ppl
        model.schema_keys = backup_keys

    print("\n=== Schema Ablation ΔPPL ===")
    for k, v in ablation_scores.items():
        print(f"{k}: {v:+.4f}")
    return ablation_scores


def _estimate_perplexity(model, tokenizer, text: str):
    """Rough perplexity estimate via log-likelihood over the prompt."""
    toks = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([toks], dtype=torch.long, device=next(model.parameters()).device)
    logits, *_ = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
    return float(torch.exp(loss))


# -----------------------------------------------
# CLI
# -----------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Explainable generation and schema analysis")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_len", type=int, default=60)
    args = p.parse_args()

    model = SchemaAugmentedSOMT.from_pretrained(args.model_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    result = generate_with_schema_attribution(model, tokenizer, args.prompt, max_len=args.max_len)
    print("\n🧠 Generated Text:\n", result["text"])

    schema_ablation_test(model, tokenizer, args.prompt, max_len=args.max_len)
