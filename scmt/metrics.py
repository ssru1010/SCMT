# metrics.py
import math
import numpy as np
from collections import Counter

def perplexity_from_loss(loss):
    return float(math.exp(loss))

# SELFIES validity placeholder: user must provide validator or use selfies lib
def is_valid_selfies(selfies_str):
    # Placeholder: integrate your SELFIES validator here.
    # e.g., import selfies; try: selfies.decoder(selfies_str) -> if works return True
    try:
        import selfies as sf
        _ = sf.decoder(selfies_str.replace(" ", ""))
        return True
    except Exception:
        return False

def novelty_uniqueness(generated_list, training_set):
    # generated_list: list of strings
    uniq = list(set(generated_list))
    uniqueness = len(uniq) / max(1, len(generated_list))
    novel = [g for g in uniq if g not in training_set]
    novelty = len(novel) / max(1, len(uniq))
    return {"uniqueness": uniqueness, "novelty": novelty, "unique_count": len(uniq)}

def schema_usage_stats(schema_keys_buffer, schema_utility_sum, schema_usage_count):
    # Accept tensors or numpy arrays
    import numpy as np
    util = np.array(schema_utility_sum).astype(float)
    counts = np.array(schema_usage_count).astype(float)
    avg_util = util / (counts + 1e-8)
    return {"utility": util.tolist(), "usage": counts.tolist(), "avg_util": avg_util.tolist()}
