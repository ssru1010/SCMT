#!/usr/bin/env python
# ====================================================
# comparison_mol.py — A/B Benchmark: Schema-SOMT vs GPT-2 (Unified Trainer, seeded + plotting)
# ====================================================
import os, subprocess, json, time, argparse, math, torch
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Helper: run subprocess safely
# ----------------------------
def run_train(model_type, run_id, base_seed=2025, extra_args=""):
    ckpt_dir = f"./checkpoints/{model_type}_mol_run_{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    seed = base_seed + run_id
    cmd = f"python train_mol_unified.py --model-type {model_type} --save-dir {ckpt_dir} --seed {seed} {extra_args}"
    print(f"\n🚀 Running {model_type.upper()} (run {run_id}, seed={seed})")
    print(f"CMD: {cmd}")
    rc = subprocess.run(cmd, shell=True)
    return rc.returncode, ckpt_dir, seed

# ----------------------------
# Helper: read metrics.json output
# ----------------------------
def read_metrics(ckpt_dir):
    metrics_path = os.path.join(ckpt_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return {"loss": None, "ppl": None, "split": None}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("eval_loss"): 
            return {"loss": None, "ppl": None, "split": data.get("split_info")}
        mean_loss = sum(data["eval_loss"]) / len(data["eval_loss"])
        ppl = math.exp(mean_loss)
        return {"loss": mean_loss, "ppl": ppl, "split": data.get("split_info")}
    except Exception as e:
        print(f"⚠️ Failed to parse metrics: {metrics_path}: {e}")
        return {"loss": None, "ppl": None, "split": None}

# ----------------------------
# Helper: count model parameters
# ----------------------------
def count_parameters(model_file):
    try:
        state = torch.load(model_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return sum(p.numel() for p in state.values())
    except Exception:
        return 0

# ----------------------------
# Plotting utility
# ----------------------------
def plot_comparison(results):
    plt.figure(figsize=(6,4))
    labels, means, stds = [], [], []

    for model_name, entries in results.items():
        ppls = [e["ppl"] for e in entries if e["ppl"] is not None]
        if not ppls:
            continue
        labels.append(model_name.upper())
        means.append(np.mean(ppls))
        stds.append(np.std(ppls))

    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    plt.xticks(x, labels)
    plt.ylabel("Perplexity (↓ better)")
    plt.title("Molecular SELFIES Benchmark\nMean ± Std over Runs")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main comparison entry
# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare ChemSOMT vs GPT-2 on molecular SELFIES")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--extra-args", type=str, default="--epochs 1 --seq-len 90 --batch-size 8 --lr 2e-4 --data ./data/test.csv")
    p.add_argument("--no-gpt2", action="store_true", help="Skip GPT-2 baseline")
    p.add_argument("--base-seed", type=int, default=2025)
    args = p.parse_args()

    results = {"somt": [], "gpt2": []}
    total_params = {}

    # (1) Schema-Augmented SOMT
    for i in range(args.runs):
        rc, ckpt, seed = run_train("somt", i, args.base_seed, args.extra_args)
        res = read_metrics(ckpt)
        results["somt"].append(res)
        model_path = os.path.join(ckpt, "pytorch_model.bin")
        if os.path.exists(model_path) and "somt" not in total_params:
            total_params["somt"] = count_parameters(model_path)
        if res["split"]:
            print(f"   🧩 Split (SOMT run {i}): {res['split']}")
        time.sleep(1)

    # (2) GPT-2 baseline
    if not args.no_gpt2:
        for i in range(args.runs):
            rc, ckpt, seed = run_train("gpt2", i, args.base_seed, args.extra_args)
            res = read_metrics(ckpt)
            results["gpt2"].append(res)
            model_path = os.path.join(ckpt, "pytorch_model.bin")
            if os.path.exists(model_path) and "gpt2" not in total_params:
                total_params["gpt2"] = count_parameters(model_path)
            if res["split"]:
                print(f"   🧩 Split (GPT2 run {i}): {res['split']}")
            time.sleep(1)

    # ----------------------------
    # Summarize and save
    # ----------------------------
    out_file = "mol_ab_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": results, "params": total_params}, f, indent=2)

    print(f"\n✅ Results saved to {out_file}")
    print("📊 Parameter Counts:")
    for k, v in total_params.items():
        print(f"  {k}: {v/1e6:.2f}M params")

    print("\n📈 Summary (mean eval loss & PPL):")
    for model_name, entries in results.items():
        vals = [e["loss"] for e in entries if e["loss"] is not None]
        if not vals:
            print(f"  {model_name}: no valid results")
            continue
        mean_loss = np.mean(vals)
        mean_ppl = math.exp(mean_loss)
        std_ppl = np.std([math.exp(v) for v in vals])
        print(f"  {model_name}: mean eval loss = {mean_loss:.4f} | mean ppl = {mean_ppl:.2f} ± {std_ppl:.2f}")

    # ----------------------------
    # Plot aggregate results
    # ----------------------------
    try:
        plot_comparison(results)
    except Exception as e:
        print("⚠️ Plotting failed:", e)

    print("\n🎯 Done. Each run used deterministic seeds offset by run index.")
