#!/usr/bin/env python
# ====================================================
# comparison_mol.py — A/B Benchmark: ChemSCMT vs GPT-2
# ====================================================
import os, subprocess, json, time, argparse, math, torch

# Utility to run training subprocess safely
def run_train(script, model_type, run_id, extra_args=""):
    ckpt_dir = f"./checkpoints/{model_type}_mol_run_{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)
    cmd = f"python {script} --save-dir {ckpt_dir} {extra_args}"
    print(f"🚀 Running {model_type.upper()} (run {run_id})")
    print(f"CMD: {cmd}")
    rc = subprocess.run(cmd, shell=True)
    return rc.returncode, ckpt_dir

def read_results(ckpt_dir):
    # Looks for trainer_state.json or metrics.json
    for fname in ["trainer_state.json", "metrics.json"]:
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # Prefer eval loss or perplexity if available
                    loss = data.get("best_eval_loss", data.get("eval_loss", None))
                    ppl = data.get("best_ppl", data.get("ppl", None))
                    return {"loss": loss, "ppl": ppl}
                except Exception:
                    continue
    return {"loss": None, "ppl": None}

def count_parameters(model_file):
    """Quick param count estimation for model sanity-check"""
    try:
        import torch
        state = torch.load(model_file, map_location="cpu")
        total_params = sum(p.numel() for p in state.values())
        return total_params
    except Exception:
        return 0

# ====================================================
# Main entry
# ====================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--extra-args", type=str, default="--epochs 1 --seq-len 90 --batch-size 8 --lr 2e-4")
    p.add_argument("--compare-gpt2", action="store_true", help="Include GPT-2 baseline in comparison")
    args = p.parse_args()

    results = {"chem_somt": [], "gpt2": []}
    total_params = {}

    # (1) ChemSCMT (Schema-Augmented SOMT)
    for i in range(args.runs):
        rc, ckpt = run_train("train_mol.py", "chem_somt", i, args.extra_args)
        res = read_results(ckpt)
        results["chem_somt"].append(res)
        # count parameters (only first run)
        model_path = os.path.join(ckpt, "pytorch_model.bin")
        if os.path.exists(model_path):
            total_params["chem_somt"] = count_parameters(model_path)
        time.sleep(1)

    # (2) GPT-2 baseline (if requested)
    if args.compare_gpt2:
        for i in range(args.runs):
            rc, ckpt = run_train("train_gpt2_mol.py", "gpt2", i, args.extra_args)
            res = read_results(ckpt)
            results["gpt2"].append(res)
            model_path = os.path.join(ckpt, "pytorch_model.bin")
            if os.path.exists(model_path):
                total_params["gpt2"] = count_parameters(model_path)
            time.sleep(1)

    # Summary and save
    out_file = "mol_ab_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": results, "params": total_params}, f, indent=2)

    print("\n✅ Saved results to", out_file)
    print("📊 Parameter Counts:")
    for k, v in total_params.items():
        print(f"  {k}: {v/1e6:.2f}M params")

    # Optional quick summary of performance
    for model_name, entries in results.items():
        vals = [e["loss"] for e in entries if e["loss"] is not None]
        if vals:
            mean_loss = sum(vals) / len(vals)
            mean_ppl = math.exp(mean_loss)
            print(f"  {model_name}: mean eval loss = {mean_loss:.4f} | mean ppl ≈ {mean_ppl:.2f}")
