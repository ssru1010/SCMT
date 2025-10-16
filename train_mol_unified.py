#!/usr/bin/env python
# ====================================================
# train_mol_unified.py — Unified Trainer for Molecular SELFIES (Train/Test Split)
# ====================================================

import os, math, random, argparse, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np, pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from FastChemTokenizerHF import FastChemTokenizerSelfies
from scmt.modeling_somt import SchemaAugmentedSOMT, SOMTConfig
from transformers import GPT2Config, GPT2LMHeadModel

# ----------------------------
# 0. Utility
# ----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_item(x, default=0.0):
    if x is None: return default
    if isinstance(x, torch.Tensor): return x.item()
    try: return float(x)
    except: return default

# ----------------------------
# 1. Dataset
# ----------------------------
class SelfiesDataset(Dataset):
    def __init__(self, csv_path, tokenizer, seq_len=90):
        df = pd.read_csv(csv_path)
        self.texts = df["SELFIES"].astype(str).tolist() if "SELFIES" in df.columns else df.iloc[:,0].astype(str).tolist()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        toks = self.tokenizer.encode(self.texts[idx])
        toks = toks[:self.seq_len]
        pad_id = self.tokenizer.pad_token_id  # = 2
        if len(toks) < self.seq_len:
            toks += [pad_id] * (self.seq_len - len(toks))
        return torch.tensor(toks, dtype=torch.long)  # shape: [seq_len]

# ----------------------------
# 2. Model Builders
# ----------------------------
def build_gpt2(vocab_size, d_model=256, nhead=8, num_layers=4, max_len=256):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=d_model, n_head=nhead, n_layer=num_layers,
        n_positions=max_len,
        bos_token_id=0, eos_token_id=1, pad_token_id=2
    )
    return GPT2LMHeadModel(cfg)

def build_somt(vocab_size, d_model=256, nhead=8, num_layers=4, max_len=256, mem_size=128):
    cfg = SOMTConfig(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                     num_layers=num_layers, max_len=max_len, mem_size=mem_size)
    return SchemaAugmentedSOMT(cfg.vocab_size, d_model=cfg.d_model, nhead=cfg.nhead,
                               num_layers=cfg.num_layers, max_len=cfg.max_len,
                               mem_size=cfg.mem_size)

# ----------------------------
# 3. Evaluation
# ----------------------------
@torch.no_grad()
# --- Inside evaluate() ---
@torch.no_grad()
def evaluate(model, loader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0
    for x in loader:
        x = x.to(device)

        if hasattr(model, "schema_keys"):  # somt
            logits, *_ = model(x)
        else:
            out = model(x, labels=None)  # don't pass labels
            logits = out.logits

        # Shift for autoregressive prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = x[:, 1:].contiguous()

        loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)


# ----------------------------
# 4. Training Loop
# ----------------------------
from torch.cuda.amp import autocast, GradScaler

from torch.cuda.amp import autocast, GradScaler

def train(model, tokenizer, train_loader, test_loader, device, epochs=1, lr=2e-4, model_type="somt"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    metrics = {k: [] for k in [
        "train_loss", "eval_loss", "ppl",
        "lm_loss", "schema_utility", "importance_entropy", "importance_l2", "budget", "routing_entropy"
    ]}

    for ep in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)

        for batch in pbar:
            x = batch.to(device)
            opt.zero_grad()

            with autocast():
                if model_type == "somt":
                    logits, _, _, _, aux = model(x)
                else:
                    out = model(x, labels=None)  # no labels; manual loss
                    logits = out.logits

                # Shift for autoregressive loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = x[:, 1:].contiguous()

                loss = criterion(
                    shift_logits.view(-1, model.config.vocab_size),
                    shift_labels.view(-1)
                )

                if model_type == "somt":
                    total_aux = (
                        aux["importance_entropy"] +
                        aux["importance_l2"] +
                        aux["schema_utility"]
                    )
                    total_loss_combined = loss + total_aux
                else:
                    total_loss_combined = loss

            scaler.scale(total_loss_combined).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

            # Track metrics
            metrics["lm_loss"].append(loss.item())
            if model_type == "somt":
                metrics["schema_utility"].append(safe_item(aux.get("schema_utility")))
                metrics["importance_entropy"].append(safe_item(aux.get("importance_entropy")))
                metrics["importance_l2"].append(safe_item(aux.get("importance_l2")))

                if hasattr(model, "budget_controller"):
                    with torch.no_grad():
                        bctx = torch.randn(1, model.d_model, device=device)
                        try:
                            metrics["budget"].append(model.budget_controller(bctx).mean().item())
                        except:
                            pass

                if hasattr(model, "schema_router"):
                    with torch.no_grad():
                        dummy = torch.randn(4, 8, model.d_model, device=device)
                        try:
                            probs = F.softmax(model.schema_router(dummy), dim=-1)
                            ent = (-probs * torch.log(probs + 1e-12)).sum(-1).mean().item()
                            metrics["routing_entropy"].append(ent)
                        except:
                            pass

        tr_loss = total_loss / len(train_loader)
        ev_loss, ppl = evaluate(
            model, test_loader, criterion, device,
            model.config.vocab_size if model_type == "somt" else model.config.vocab_size
        )

        metrics["train_loss"].append(tr_loss)
        metrics["eval_loss"].append(ev_loss)
        metrics["ppl"].append(ppl)
        print(f"Epoch {ep+1}: train {tr_loss:.3f} | eval {ev_loss:.3f} | ppl {ppl:.2f}")

    return model, metrics


# ----------------------------
# 5. Main Entry
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Unified Trainer for GPT-2 vs Schema-Augmented SOMT on SELFIES (with train/test split)")
    p.add_argument("--model-type", choices=["somt","gpt2"], default="somt")
    p.add_argument("--data", type=str, default="./data/test.csv")
    p.add_argument("--seq-len", type=int, default=90)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-dir", type=str, default="./checkpoints/mol_unified")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--mem-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()

    # Use the provided seed (must be after parse_args)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using {device} | Model={args.model_type} | seed={args.seed}")

    tokenizer = FastChemTokenizerSelfies.from_pretrained("./tokenizer_vocab/selftok_reordered")
    full_ds = SelfiesDataset(args.data, tokenizer, args.seq_len)

    # 90/10 train/test split (deterministic via args.seed)
    train_size = int(0.9 * len(full_ds))
    test_size = len(full_ds) - train_size
    g = torch.Generator().manual_seed(args.seed)
    train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size*2, shuffle=False)
    vocab_size = tokenizer.vocab_size
    print(f"📚 Dataset: {len(full_ds)} samples → Train: {train_size}, Test: {test_size} | Vocab: {vocab_size}")

    if args.model_type == "somt":
        model = build_somt(vocab_size, args.d_model, args.nhead, args.num_layers, args.seq_len, args.mem_size)
    else:
        model = build_gpt2(vocab_size, args.d_model, args.nhead, args.num_layers, args.seq_len)
    model.to(device)
    print(f"🧠 Model built with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    model, metrics = train(model, tokenizer, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr, model_type=args.model_type)

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)

    # record split info and seed for reproducibility
    metrics["split_info"] = {"train_size": len(train_ds), "test_size": len(test_ds), "seed": args.seed}
    with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved → {args.save_dir}/metrics.json")

# Plot training curves
    plt.figure(figsize=(8,5))
    plt.plot(metrics["lm_loss"], label="LM Loss")
    if args.model_type == "somt":
        plt.plot(metrics["schema_utility"], label="Schema Utility")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    # Save instead of show
        # Save instead of show
    plot_path = os.path.join(
        args.save_dir,
        f"training_plot_{args.model_type}_{args.seed}.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Training plot saved → {plot_path}")


if __name__ == "__main__":
    main()
