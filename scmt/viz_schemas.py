#!/usr/bin/env python
# ====================================================
# viz_schemas.py — Schema Visualization Utility
# ====================================================
# Dependencies: torch, pandas, jinja2 (for HTML export)
# Usage:
#   python viz_schemas.py --model ./checkpoints/somt --format html --top_k 10
# ====================================================

import os, json, argparse
import torch
import pandas as pd
import torch.nn.functional as F
from modeling_somt import SchemaAugmentedSOMT, SOMTConfig
from modeling_utils import visualize_schemas as _base_visualize

# -----------------------------------------------
# Utility: Mask common/stop tokens
# -----------------------------------------------
STOP_TOKENS = {".", ",", "the", "a", "an", "to", "and", "[C]", "[PAD]", ""}

def _mask_tokens(tokens):
    return [t for t in tokens if t not in STOP_TOKENS]

# -----------------------------------------------
# Enhanced visualization
# -----------------------------------------------
@torch.no_grad()
def visualize_schemas(
    model,
    tokenizer,
    top_k: int = 10,
    key_space: bool = True,
    mask_stop: bool = False,
    export: str = None,     # "html", "latex", or None
    export_path: str = "schemas_viz.html",
):
    """
    Enhanced schema visualization for interpretability.
    Can optionally mask stop tokens and export as HTML/LaTeX table.
    """
    model.eval()
    device = next(model.parameters()).device

    # Raw embeddings projected into key-space (default)
    embed_matrix = model.embed.weight.data  # (V, D)
    projected_tokens = model.key_proj(embed_matrix) if key_space else embed_matrix

    schema_norm = F.normalize(model.schema_keys, p=2, dim=-1)
    token_norm = F.normalize(projected_tokens, p=2, dim=-1)
    sim = torch.mm(schema_norm, token_norm.T)

    rows = []
    for i in range(sim.size(0)):
        top_vals, top_ids = torch.topk(sim[i], k=top_k, largest=True)
        tokens = [tokenizer.decode([tid.item()]).strip() for tid in top_ids]
        if mask_stop:
            tokens = _mask_tokens(tokens)
        pairs = [(tok, float(val)) for tok, val in zip(tokens, top_vals)]
        row = {"schema_id": i, "tokens": tokens, "scores": [round(v, 3) for v in top_vals.tolist()]}
        rows.append(row)

        print(f"Schema {i:02d}: " + " | ".join(f"{tok} ({val:.3f})" for tok, val in pairs))

    # Optional export
    if export:
        df = pd.DataFrame(rows)
        if export == "html":
            html = df.to_html(index=False, escape=False)
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"\n✅ Exported to {export_path}")
        elif export == "latex":
            tex = df.to_latex(index=False, escape=False)
            with open(export_path.replace(".html", ".tex"), "w", encoding="utf-8") as f:
                f.write(tex)
            print(f"\n✅ Exported to {export_path.replace('.html', '.tex')}")
        else:
            print("⚠️ Unknown export format; skipping export.")

# -----------------------------------------------
# CLI
# -----------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Schema visualization tool")
    p.add_argument("--model_dir", type=str, required=True, help="Path to trained SOMT checkpoint")
    p.add_argument("--format", type=str, choices=["html", "latex", "none"], default="none")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--mask_stop", action="store_true")
    p.add_argument("--no_key_space", action="store_true")
    args = p.parse_args()

    # Load model
    model = SchemaAugmentedSOMT.from_pretrained(args.model_dir)
    from transformers import AutoTokenizer  # assuming HF-style tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    visualize_schemas(
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        key_space=not args.no_key_space,
        mask_stop=args.mask_stop,
        export=None if args.format == "none" else args.format,
        export_path=f"{args.model_dir}/schemas_viz.{args.format if args.format != 'none' else 'html'}"
    )
