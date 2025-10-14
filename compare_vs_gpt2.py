# compare_vs_gpt2.py
#!/usr/bin/env python
import subprocess, json, os, argparse, time

def run_train(model_type, run_id, extra_args=""):
    ckpt = f"./checkpoints/{model_type}_run_{run_id}"
    cmd = f"python train.py --model-type {model_type} --checkpoint-dir {ckpt} {extra_args}"
    print("RUN:", cmd)
    r = subprocess.run(cmd, shell=True)
    return r.returncode

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--extra-args", type=str, default="--epochs 1 --fraction 0.00001")
    args = p.parse_args()

    results = {"somt": [], "gpt2": []}
    for i in range(args.runs):
        for mt in ["somt", "gpt2"]:
            rc = run_train(mt, i, args.extra_args)
            # read trainer_state.json
            ckpt = f"./checkpoints/{mt}_run_{i}"
            tsf = os.path.join(ckpt, "trainer_state.json")
            if os.path.exists(tsf):
                with open(tsf, "r", encoding="utf-8") as f:
                    ts = json.load(f)
                results[mt].append(ts.get("best_eval_loss"))
            else:
                results[mt].append(None)
            time.sleep(1)
    with open("ab_compare_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved ab_compare_results.json")
