#!/usr/bin/env python
# ====================================================
# ChemSCMT — Schema-Augmented SOMT (W) Molecular Model
# ====================================================

import os, math, time, random, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from FastChemTokenizerHF import FastChemTokenizerSelfies
from scmt import SchemaAugmentedSOMT, SOMTConfig

# ====================================================
# 0. Utility
# ====================================================

def set_seed(seed: int = 2025):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_item(d, key, default=0.0):
    v = d.get(key, None)
    if v is None: return default
    if isinstance(v, torch.Tensor): return v.item()
    try: return float(v)
    except Exception: return default

# ====================================================
# 1. Dataset
# ====================================================

class SelfiesDataset(Dataset):
    def __init__(self, csv_path, tokenizer, seq_len=90):
        df = pd.read_csv(csv_path)
        self.texts = df["SELFIES"].astype(str).tolist() if "SELFIES" in df.columns else df.iloc[:, 0].astype(str).tolist()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        toks = self.tokenizer.encode(self.texts[idx])
        toks = toks[:self.seq_len]
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        if len(toks) < self.seq_len:
            toks += [pad_id] * (self.seq_len - len(toks))
        x = torch.tensor(toks, dtype=torch.long)
        y = torch.roll(x, shifts=-1, dims=0)
        return x, y

# ====================================================
# 2. Model Setup
# ====================================================

def build_model(vocab_size: int, d_model=256, nhead=8, num_layers=4, max_len=256, mem_size=128):
    cfg = SOMTConfig(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                     num_layers=num_layers, max_len=max_len, mem_size=mem_size)
    model = SchemaAugmentedSOMT(cfg.vocab_size, d_model=cfg.d_model, nhead=cfg.nhead,
                                num_layers=cfg.num_layers, max_len=cfg.max_len,
                                mem_size=cfg.mem_size)
    return model

# ====================================================
# 3. Training & Evaluation
# ====================================================

def evaluate(model, loader, criterion, device, vocab_size):
    model.eval(); total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, *_ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)

def train(model, train_loader, test_loader, tokenizer, epochs=1, lr=2e-4, device="cpu"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    history = {"train_loss": [], "eval_loss": [], "ppl": []}
    metrics = {k: [] for k in ["budget", "schema_util", "routing_entropy", "lm_loss", "schema_loss", "entropy_reg", "importance_l2"]}

    for ep in range(epochs):
        model.train(); total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _, _, _, aux = model(x)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            total_aux = sum(aux.values())
            (loss + total_aux).backward()
            opt.step()

            with torch.no_grad():
                metrics["lm_loss"].append(loss.item())
                metrics["schema_loss"].append(safe_item(aux, "schema_utility"))
                metrics["entropy_reg"].append(safe_item(aux, "importance_entropy"))
                metrics["importance_l2"].append(safe_item(aux, "importance_l2"))
                metrics["schema_util"].append(safe_item(aux, "schema_utility"))

                if hasattr(model, "budget_controller"):
                    dummy = torch.randn(1, model.d_model, device=device)
                    try:
                        budget = model.budget_controller(dummy).mean().item()
                        metrics["budget"].append(budget)
                    except Exception: pass

                if hasattr(model, "schema_router"):
                    dummy_keys = torch.randn(4, 8, model.d_model, device=device)
                    try:
                        probs = F.softmax(model.schema_router(dummy_keys), dim=-1)
                        entropy = (-probs * torch.log(probs + 1e-12)).sum(-1).mean().item()
                        metrics["routing_entropy"].append(entropy)
                    except Exception: pass

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        tr_loss = total_loss / len(train_loader)
        ev_loss, ppl = evaluate(model, test_loader, criterion, device, model.vocab_size)
        history["train_loss"].append(tr_loss)
        history["eval_loss"].append(ev_loss)
        history["ppl"].append(ppl)
        print(f"Epoch {ep+1}: train {tr_loss:.3f} | eval {ev_loss:.3f} | ppl {ppl:.2f}")

    return model, history, metrics

# ====================================================
# 4. Generation
# ====================================================

@torch.no_grad()
def generate(model, tokenizer, prompt="", max_len=80, temperature=0.8, device="cpu"):
    model.eval()
    start_id = getattr(tokenizer, "bos_token_id", getattr(tokenizer, "pad_token_id", 0))
    if prompt == "":
        input_ids = torch.tensor([[start_id]], dtype=torch.long).to(device)
    else:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(input_ids, max_length=max_len, temperature=temperature,
                         eos_token_id=getattr(tokenizer, "eos_token_id", None),
                         pad_token_id=getattr(tokenizer, "pad_token_id", None))
    out_ids = out[0].tolist() if isinstance(out, torch.Tensor) else out[0]
    return tokenizer.decode(out_ids, skip_special_tokens=True)

# ====================================================
# 5. Main Entry
# ====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/test.csv")
    parser.add_argument("--seq-len", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/somt_mol_test")
    args = parser.parse_args()

    set_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using {device}")

    # Tokenizer & dataset
    tokenizer = FastChemTokenizerSelfies.from_pretrained("./tokenizer_vocab/selftok_reordered")
    ds = SelfiesDataset(args.data, tokenizer, args.seq_len)
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(ds, batch_size=args.batch_size*2, shuffle=False)
    vocab_size = tokenizer.vocab_size
    print(f"📚 Dataset: {len(ds)} samples | Vocab size: {vocab_size}")

    # Model
    model = build_model(vocab_size).to(device)
    model.vocab_size = vocab_size  # convenience
    print("🧠 Model built.")

    # Train
    model, hist, metrics = train(model, train_loader, test_loader, tokenizer, epochs=args.epochs, lr=args.lr, device=device)

    # Save + reload
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    model_loaded = SchemaAugmentedSOMT.from_pretrained(args.save_dir).to(device)
    print("✅ Reloaded successfully.")

    # Generation sanity test
    print("🧪 Generation Samples:")
    for i in range(2):
        print(f"--- Sample {i+1} ---")
        print(generate(model_loaded, tokenizer, prompt="", max_len=60, temperature=0.8, device=device))

    # Plot
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.plot(metrics["budget"]); plt.title("Budget Utilization")
    plt.subplot(2,2,2); plt.plot(metrics["schema_util"]); plt.title("Schema Utility")
    plt.subplot(2,2,3); plt.plot(metrics["routing_entropy"]); plt.title("Routing Entropy")
    plt.subplot(2,2,4)
    plt.plot(metrics["lm_loss"], label="LM Loss")
    plt.plot(metrics["schema_loss"], label="Schema Loss")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
