# train.py
#!/usr/bin/env python
import argparse, json, os, time
from pathlib import Path
from datetime import datetime
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# If you have the modular somt package, import it; otherwise the script will try to use the inline GPT2 path.
try:
    from scmt import SchemaAugmentedSOMT, SOMTConfig
    SOMT_AVAILABLE = True
except Exception:
    SOMT_AVAILABLE = False

# Optional GPT-2 baseline
try:
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ---- Simple streaming dataset used for TinyStories (line-per-example) ----
class StreamingTextDataset(Dataset):
    def __init__(self, txt_path, tokenizer, seq_len=128, fraction=1.0):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        sample_size = max(1, int(len(lines) * fraction))
        self.texts = random.sample(lines, sample_size)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        toks = self.tokenizer.encode(self.texts[idx])
        toks = toks[: self.seq_len]
        if len(toks) < self.seq_len:
            toks += [self.tokenizer.pad_token_id] * (self.seq_len - len(toks))
        x = torch.tensor(toks, dtype=torch.long)
        y = torch.roll(x, shifts=-1, dims=0)
        return x, y

# ---- Helpers: save/load trainer state + pretty logging ----
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def save_checkpoint(save_dir, model, optimizer, scheduler, trainer_state):
    ensure_dir(save_dir)
    model.save_pretrained(save_dir) if hasattr(model, "save_pretrained") else torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    # save optimizer + trainer state
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": getattr(scheduler, "state_dict", lambda: None)()}, os.path.join(save_dir, "optim.pt"))
    with open(os.path.join(save_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)
    print(f"[{now()}] ✅ checkpoint saved -> {save_dir}")

def load_checkpoint(load_dir, model, optimizer=None, scheduler=None, map_location=None):
    # load model
    if hasattr(model, "from_pretrained") and os.path.exists(os.path.join(load_dir, "pytorch_model.bin")):
        model_loaded = model.__class__.from_pretrained(load_dir, map_location=map_location)
        model.load_state_dict(model_loaded.state_dict())
    elif os.path.exists(os.path.join(load_dir, "pytorch_model.bin")):
        sd = torch.load(os.path.join(load_dir, "pytorch_model.bin"), map_location=map_location)
        model.load_state_dict(sd)
    else:
        raise FileNotFoundError("pytorch_model.bin not found in checkpoint dir")

    # load optimizer
    if optimizer is not None and os.path.exists(os.path.join(load_dir, "optim.pt")):
        d = torch.load(os.path.join(load_dir, "optim.pt"), map_location=map_location)
        optimizer.load_state_dict(d["optimizer"])
    trainer_state = {}
    if os.path.exists(os.path.join(load_dir, "trainer_state.json")):
        with open(os.path.join(load_dir, "trainer_state.json"), "r", encoding="utf-8") as f:
            trainer_state = json.load(f)
    print(f"[{now()}] ✅ checkpoint loaded from {load_dir}")
    return trainer_state

# ---- Main training routine ----
def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[{now()}] 🚀 Using device: {device}")

    # tokenizer + datasets
    if args.model_type == "gpt2":
        assert HF_AVAILABLE, "Install transformers to use gpt2 baseline."
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_ds = StreamingTextDataset(args.train_data, tokenizer, seq_len=args.seq_len, fraction=args.fraction)
        test_ds  = StreamingTextDataset(args.valid_data, tokenizer, seq_len=args.seq_len, fraction=1.0)
        vocab_size = tokenizer.vocab_size
    else:
        # somt
        assert SOMT_AVAILABLE, "somt package not importable. Ensure somt package is in PYTHONPATH."
        # we still use GPT2 tokenizer for TinyStories by default for tokenization parity
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_ds = StreamingTextDataset(args.train_data, tokenizer, seq_len=args.seq_len, fraction=args.fraction)
        test_ds  = StreamingTextDataset(args.valid_data, tokenizer, seq_len=args.seq_len, fraction=0.1)
        vocab_size = tokenizer.vocab_size

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)
    print(f"[{now()}] 📚 Train samples: {len(train_ds)} | Vocab: {vocab_size}")

    # build model
    if args.model_type == "gpt2":
        cfg = GPT2Config(vocab_size=vocab_size, n_layer=args.num_layers, n_head=args.nhead, n_embd=args.d_model)
        model = GPT2LMHeadModel(cfg).to(device)
    else:
        cfg = SOMTConfig(vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, max_len=args.seq_len, mem_size=args.mem_size)
        model = SchemaAugmentedSOMT(cfg.vocab_size, d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers, max_len=cfg.max_len, mem_size=cfg.mem_size).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = None

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    start_epoch = 0
    best_eval = float("inf")
    trainer_state = {"epoch": 0, "step": 0, "best_eval_loss": None}

    # resume
    if args.resume:
        if os.path.exists(args.checkpoint_dir):
            ts = load_checkpoint(args.checkpoint_dir, model, optimizer=opt, scheduler=scheduler, map_location=device)
            start_epoch = ts.get("epoch", 0)
            trainer_state.update(ts)
            best_eval = ts.get("best_eval_loss", float("inf"))
        else:
            print(f"[{now()}] No checkpoint found at {args.checkpoint_dir} (resume ignored)")

    metrics_log = {"lm_loss": [], "importance_entropy": [], "importance_l2": [], "schema_utility": [], "budget": [], "routing_entropy": []}

    for ep in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epochs}", leave=False)
        for step, (x, y) in enumerate(pbar):
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits, updated_keys, updated_vals, updated_age, aux = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_aux = aux["importance_entropy"] + aux["importance_l2"] + aux["schema_utility"]
            (loss + total_aux).backward()
            opt.step()

            epoch_loss += loss.item()
            trainer_state["step"] += 1

            # track simple metrics (append scalars)
            metrics_log["lm_loss"].append(loss.item())
            metrics_log["importance_entropy"].append(aux["importance_entropy"].item())
            metrics_log["importance_l2"].append(aux["importance_l2"].item())
            metrics_log["schema_utility"].append(aux["schema_utility"].item())

            # budget (use budget_controller on batch mean)
            if hasattr(model, "budget_controller"):
                with torch.no_grad():
                    bctx = logits.new_zeros((1, model.d_model))
                    budget = model.budget_controller(bctx).mean().item()
                    metrics_log["budget"].append(budget)

            # routing entropy: if we have schemas formed, approximate via _form_schemas
            if hasattr(model, "_form_schemas"):
                with torch.no_grad():
                    if updated_keys is not None and updated_keys.size(1) > 0:
                        # form schemas for peeked batch
                        batch_schema_keys, routing_probs = model._form_schemas(updated_keys, updated_vals)
                        routing_entropy = - (routing_probs * torch.log(routing_probs + 1e-12)).sum(dim=-1).mean().item()
                        metrics_log["routing_entropy"].append(routing_entropy)

            if (trainer_state["step"] % args.log_every) == 0:
                print(f"[{now()}] ep {ep+1} step {trainer_state['step']}: loss={loss.item():.4f} aux_schema={aux['schema_utility'].item():.6f}")

        # eval
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device); y = y.to(device)
                logits, *_ = model(x)
                l = criterion(logits.view(-1, vocab_size), y.view(-1))
                total_loss += l.item()
        avg_eval_loss = total_loss / len(test_loader)
        ppl = float(math.exp(avg_eval_loss))
        print(f"[{now()}] EPOCH {ep+1} eval_loss={avg_eval_loss:.4f} ppl={ppl:.2f}")

        # checkpoint best
        trainer_state["epoch"] = ep+1
        trainer_state["best_eval_loss"] = min(best_eval, avg_eval_loss)
        if avg_eval_loss < best_eval:
            best_eval = avg_eval_loss
            trainer_state["best_eval_loss"] = best_eval
            save_checkpoint(args.checkpoint_dir, model, opt, scheduler, trainer_state)

        # save metrics log snapshot
        ensure_dir(args.checkpoint_dir)
        with open(os.path.join(args.checkpoint_dir, "metrics_log.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_log, f, indent=2)

    print(f"[{now()}] Training finished.")
    return model, metrics_log

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=str, default="./TinyStories-train.csv")
    p.add_argument("--valid-data", type=str, default="./TinyStories-valid.csv")
    p.add_argument("--model-type", choices=["somt", "gpt2"], default="somt")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints/somt_run")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=128)
    p.add_argument("--d-model", dest="d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", dest="num_layers", type=int, default=4)
    p.add_argument("--mem-size", dest="mem_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fraction", type=float, default=0.00001)
    p.add_argument("--cpu", action="store_true", help="force cpu")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model, metrics = train(args)
    # quick plot of lm_loss to verify (shows when run interactively)
    try:
        plt.figure(figsize=(6,3))
        plt.plot(metrics["lm_loss"])
        plt.title("LM loss (steps)")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass
