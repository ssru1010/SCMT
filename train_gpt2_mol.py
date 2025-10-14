#!/usr/bin/env python
# ====================================================
# train_gpt2_mol.py — GPT-2 Baseline for Molecular SELFIES
# ====================================================

import os, math, time, random, argparse
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd, numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import GPT2Config, GPT2LMHeadModel

from FastChemTokenizerHF import FastChemTokenizerSelfies

# ====================================================
# 0. Utility
# ====================================================

def set_seed(seed: int = 2025):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        toks = toks[: self.seq_len]
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        if len(toks) < self.seq_len:
            toks += [pad_id] * (self.seq_len - len(toks))
        x = torch.tensor(toks, dtype=torch.long)
        y = torch.roll(x, shifts=-1, dims=0)
        return x, y

# ====================================================
# 2. Model setup (parameter-matched GPT-2)
# ====================================================

def build_gpt2(vocab_size: int, d_model=256, nhead=8, num_layers=4, max_len=256):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=d_model,
        n_head=nhead,
        n_layer=num_layers,
        n_positions=max_len,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0
    )
    model = GPT2LMHeadModel(cfg)
    return model

# ====================================================
# 3. Training & Evaluation
# ====================================================

def evaluate(model, loader, criterion, device, vocab_size):
    model.eval(); total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, labels=y)
            total_loss += out.loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)

def train(model, train_loader, test_loader, tokenizer, epochs=1, lr=2e-4, device="cpu"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    history = {"train_loss": [], "eval_loss": [], "ppl": []}
    metrics = {"lm_loss": []}

    for ep in range(epochs):
        model.train(); total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x, labels=y)
            loss = out.loss
            loss.backward()
            opt.step()
            metrics["lm_loss"].append(loss.item())
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        tr_loss = total_loss / len(train_loader)
        ev_loss, ppl = evaluate(model, test_loader, criterion, device, model.config.vocab_size)
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
    out = model.generate(
        input_ids,
        max_length=max_len,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )
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
    parser.add_argument("--save-dir", type=str, default="./checkpoints/gpt2_mol_test")
    args = parser.parse_args()

    set_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using {device}")

    # Tokenizer & Dataset
    tokenizer = FastChemTokenizerSelfies.from_pretrained("./tokenizer_vocab/selftok_reordered")
    ds = SelfiesDataset(args.data, tokenizer, args.seq_len)
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(ds, batch_size=args.batch_size*2, shuffle=False)
    vocab_size = tokenizer.vocab_size
    print(f"📚 Dataset: {len(ds)} samples | Vocab size: {vocab_size}")

    # Model
    model = build_gpt2(vocab_size).to(device)
    print("🧠 GPT-2 Baseline built.")

    # Train
    model, hist, metrics = train(model, train_loader, test_loader, tokenizer, epochs=args.epochs, lr=args.lr, device=device)

    # Save & reload
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    model_loaded = GPT2LMHeadModel.from_pretrained(args.save_dir).to(device)
    print("✅ Reloaded successfully.")

    # Generation sanity check
    print("🧪 Generation Samples:")
    for i in range(2):
        print(f"--- Sample {i+1} ---")
        print(generate(model_loaded, tokenizer, prompt="", max_len=60, temperature=0.8, device=device))

    # Simple plot
    plt.figure(figsize=(8, 5))
    plt.plot(metrics["lm_loss"], label="LM Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
