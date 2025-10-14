#!/usr/bin/env python
# ====================================================
# schema_analysis.py — Structural and Semantic Schema Analysis
# ====================================================
# Dependencies: torch, numpy, scikit-learn
# Usage:
#   python schema_analysis.py --model ./checkpoints/somt --top_k 10
# ====================================================

import os, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from modeling_somt import SchemaAugmentedSOMT
from modeling_utils import visualize_schemas


# -----------------------------------------------
# Cosine similarity matrix and redundancy check
# -----------------------------------------------
@torch.no_grad()
def compute_schema_similarity(model):
    schema_norm = F.normalize(model.schema_keys, p=2, dim=-1)
    sim_matrix = torch.mm(schema_norm, schema_norm.T).cpu().numpy()
    np.fill_diagonal(sim_matrix, 0.0)
    redundant = np.argwhere(sim_matrix > 0.95)
    redund_pairs = [(int(i), int(j), float(sim_matrix[i, j])) for i, j in redundant if i < j]
    print(f"🔍 Found {len(redund_pairs)} redundant schema pairs (cos>0.95)")
    for (i, j, s) in redund_pairs[:10]:
        print(f"  Schema {i} ↔ {j}: {s:.3f}")
    return sim_matrix, redund_pairs


# -----------------------------------------------
# t-SNE dimensionality reduction
# -----------------------------------------------
def reduce_embeddings_tsne(embeddings, n_components=2, perplexity=5, random_state=42):
    reducer = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=min(perplexity, max(2, embeddings.shape[0] // 2)),
        learning_rate="auto",
        init="random",
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


# -----------------------------------------------
# Semantic labeling via top tokens
# -----------------------------------------------
@torch.no_grad()
def label_schemas(model, tokenizer, top_k=8):
    embed_matrix = model.embed.weight.data
    projected = model.key_proj(embed_matrix)
    schema_norm = F.normalize(model.schema_keys, p=2, dim=-1)
    token_norm = F.normalize(projected, p=2, dim=-1)
    sim = torch.mm(schema_norm, token_norm.T)
    labels = []
    for i in range(sim.size(0)):
        _, top_ids = torch.topk(sim[i], k=top_k)
        toks = [tokenizer.decode([t.item()]).strip() for t in top_ids]
        labels.append(", ".join(toks))
    return labels


# -----------------------------------------------
# Full analysis pipeline
# -----------------------------------------------
@torch.no_grad()
def analyze_schemas(model, tokenizer, top_k=8, save_path=None):
    device = next(model.parameters()).device
    schema_keys = model.schema_keys.cpu().numpy()

    sim_matrix, redund_pairs = compute_schema_similarity(model)
    reduced = reduce_embeddings_tsne(schema_keys)
    labels = label_schemas(model, tokenizer, top_k=top_k)

    # KMeans clustering
    n_clusters = min(10, len(schema_keys))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_ids = kmeans.fit_predict(reduced)

    result = {
        "positions": reduced.tolist(),
        "labels": labels,
        "redundant_pairs": redund_pairs,
        "clusters": cluster_ids.tolist(),
    }
    print(f"💡 Avg intra-cluster cosine: {np.mean(sim_matrix):.3f}")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Saved schema analysis → {save_path}")

    # Visualization preview
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=cluster_ids, cmap="tab10", s=60, alpha=0.7
        )
        for i, _ in enumerate(labels):
            plt.annotate(f"S{i}", (reduced[i, 0], reduced[i, 1]), fontsize=6, alpha=0.7)
        plt.title("Schema Embedding Space (t-SNE)")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("⚠️ Plotting failed:", e)

    return result


# -----------------------------------------------
# CLI
# -----------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Schema clustering and semantic analysis (t-SNE based)")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--save", type=str, default="schema_analysis.json")
    args = p.parse_args()

    model = SchemaAugmentedSOMT.from_pretrained(args.model_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    analyze_schemas(model, tokenizer, top_k=args.top_k, save_path=args.save)
