# plot_dynamics.py
import json, os, argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    if "budget" in metrics and metrics["budget"]:
        plt.plot(metrics["budget"])
        plt.title("Memory budget utilization")
    else:
        plt.text(0.5,0.5,"No budget data", ha="center")
    plt.xlabel("step")

    plt.subplot(2,2,2)
    if "schema_utility" in metrics and metrics["schema_utility"]:
        util = np.array(metrics["schema_utility"])
        # EMA smoothing for readability
        alpha = 0.05
        s = []
        v = 0.0
        for x in util:
            v = alpha * x + (1-alpha) * v
            s.append(v)
        plt.plot(s)
        plt.title("Schema utility (EMA)")
    else:
        plt.text(0.5,0.5,"No schema util data", ha="center")

    plt.subplot(2,2,3)
    if "routing_entropy" in metrics and metrics["routing_entropy"]:
        plt.plot(metrics["routing_entropy"])
        plt.title("Routing entropy")
    else:
        plt.text(0.5,0.5,"No routing entropy", ha="center")

    plt.subplot(2,2,4)
    if "lm_loss" in metrics:
        plt.plot(metrics["lm_loss"], label="lm_loss", linewidth=1)
    if "schema_utility" in metrics:
        plt.plot(metrics["schema_utility"], label="schema_utility", linewidth=1)
    if "importance_entropy" in metrics:
        plt.plot(metrics["importance_entropy"], label="importance_entropy", linewidth=1)
    plt.legend(); plt.title("Loss components / regs")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, default="./checkpoints/somt_run/metrics_log.json")
    args = p.parse_args()
    if not os.path.exists(args.metrics):
        raise FileNotFoundError(args.metrics)
    plot_metrics(args.metrics)
