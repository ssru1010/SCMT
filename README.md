# Schema-Augmented Self-Optimizing Memory Transformer (SCMT)

---

## **Model Overview**

**SCMT** is an experimental transformer architecture that extends the **Self-Optimizing Memory Transformer (SOMT)** with schema-level abstraction and adaptive episodic memory management. It introduces a dual mechanism of **instance-level memory** and **schema-level generalization**, enabling the model to learn compact, reusable representational patterns over streaming or long-context data.

The implementation remains self-contained and modular, designed for research compatibility with HuggingFace-style APIs while emphasizing interpretability and experimental transparency.

---

## **Intended Use**

This model is intended for **research exploration** in:

* Adaptive, entropy-regulated memory transformers.
* Schema abstraction and hierarchical reasoning.
* Symbolic generalization across structured domains (e.g., molecules, code, or text).

It is **not suitable for production deployment** and should be used only in controlled research settings.

---

## **Architectural Summary**

| Component            | Description                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Base Encoder**     | TransformerEncoder (causal masked) for autoregressive modeling.                                                |
| **Memory Module**    | Dynamic external memory with learned importance, recency decay, and entropy-thresholded writing.               |
| **Schema Router**    | Aggregates episodic traces into schema representations (`num_schemas`), serving as generalized latent anchors. |
| **Uncertainty Gate** | Scales attention queries based on local token entropy.                                                         |
| **Retrieval Fusion** | Combines schema-level and instance-level retrievals into a unified latent space.                               |
| **Auxiliary Heads**  | Regularizers for entropy smoothness, L2 importance, and schema-utility consistency.                            |
| **Generation Head**  | Shared-weight LM head with top-k/top-p filtering and repetition penalties.                                     |

---

## **Configuration (SOMTConfig)**

| Parameter             | Default | Description                           |
| --------------------- | ------- | ------------------------------------- |
| `vocab_size`          | 50257   | Vocabulary size (GPT-2 compatible).   |
| `d_model`             | 256     | Transformer hidden size.              |
| `nhead`               | 8       | Attention heads.                      |
| `num_layers`          | 4       | Transformer layers.                   |
| `max_len`             | 256     | Context length.                       |
| `mem_size`            | 128     | Episodic memory capacity.             |
| `num_schemas`         | 32      | Number of schema units.               |
| `recency_lambda`      | 0.01    | Recency weighting.                    |
| `entropy_reg_coef`    | 1e-3    | Entropy regularization.               |
| `l2_importance_coef`  | 1e-4    | L2 regularization for importance.     |
| `schema_utility_coef` | 1e-2    | Schema utility objective coefficient. |

---

## **Design Rationale**

The design aims to capture a *dialectical balance* between **episodic specificity** and **schematic generality**.
Entropy is employed as a meta-signal guiding both **memory updates** and **retrieval modulation**, ensuring capacity is allocated adaptively according to uncertainty.
Schema abstraction acts as an emergent compression mechanism over episodic traces, promoting transferability and stability in representational dynamics.

---

## **Implementation Notes**

* Framework: **PyTorch ≥ 2.1**
* Save/load: `save_pretrained()` and `from_pretrained()`
* Generation: supports temperature, top-k/top-p, and repetition penalties
* Minimal dependencies; lightweight experimental interface
* Behavior modeled as decoder-only Transformer for autoregressive tasks

---

## **Model Behavior Summary (Preliminary Observations)**

### **Training Dynamics**

Training over ~3k steps indicates stable entropy-regulated adaptation:
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/cc7a1679-4f5e-4578-bdb5-cb330f448810" />

* **Memory Budget Utilization** fluctuates between 0.2–0.9, showing non-saturated, flexible gating.
* **Routing Entropy** remains centered near 3.0, indicating sustained schema diversity.
* **Loss Components** exhibit monotonic decline in LM loss; auxiliary regularizers remain near steady baseline.
* **Schema Utility (EMA)** converges rapidly, consistent with early stabilization of schema representations.

### **Qualitative Schema Analysis**

**Chemical sequence modeling (24 k samples, vocab 793)**:
Schemas self-organize around interpretable substructures such as `[C]`, `[N][C][C][C][C]`, `[O][C][Branch1]`, suggesting emergent recognition of chemical motifs.

```text
Schema  0: [C] (0.576) | [N] [C] [C] [C] [C] (0.539) | [O] [C] [Branch1] (0.511) | [C] [C] [C] [Branch1] (0.497) | [N] [C] [C] (0.488) | [N] (0.480) | [N] [C] (0.471) | [N] [C] [C] [C] (0.465)
Schema  1: [C] (0.534) | [N] [C] [C] [C] [C] (0.517) | [#C] (0.464) | [N] [C] (0.435) | [=N] (0.433) | [Ring2] (0.431) | [C] [C] [C] [Branch1] (0.423) | [O] [C] [C] [C] [C] (0.419)
Schema  2: [N] [C] [C] [C] [C] (0.487) | [C] (0.477) | [O] [C] [C] [C] [C] (0.425) | [C] [C] [C] [Branch1] (0.410) | [N] [C] [=Branch1] [C] [=O] (0.397) | [N] [C] (0.391) | [N] [C] [C] (0.390) | [=Branch1] (0.372)
Schema  3: [C] (0.573) | [N] [C] [C] [C] [C] (0.535) | [Ring2] (0.515) | [#C] (0.485) | [C] [O] [C] (0.480) | [N] (0.477) | [=N] (0.475) | [N] [C] (0.465)
Schema  4: [N] (0.549) | [C] (0.514) | <s> (0.511) | [O] [C] [Branch1] (0.495) | [Branch1] (0.495) | [C] [=C] [C] [=C] (0.491) | [O] [C] [C] [=C] (0.490) | [N] [C] [C] [C] (0.461)
Schema  5: [N] [C] [C] [C] [C] (0.482) | [C] (0.421) | [O] [C] [C] [C] [C] (0.410) | [N] [C] [=Branch1] [C] [=O] (0.379) | [C] [C] [C] [Branch1] (0.368) | [N] [C] [C] (0.358) | [N] [C] (0.352) | [N] [C] [=Branch1] [C] [=O] [C] (0.339)
Schema  6: [C] (0.500) | [C] [O] [C] (0.495) | [N] (0.491) | [O] [C] [Branch1] (0.481) | [C] [C] [C] [C] [C] (0.475) | [Branch1] (0.473) | [F] (0.463) | [C] [C] [N] (0.462)
```

**Natural language modeling (13 k samples, vocab 50 k)**:
Schemas cluster on punctuation and short-term syntactic markers (e.g., `"`, `.`, `!`, `again`, `smiled`, `sad`), reflecting early grammatical organization but limited long-range semantic control.

```text
Schema  0: " (0.487) | ! (0.378) |  again (0.365) |  from (0.347) | ? (0.322) | ." (0.319) | ?" (0.319) |  food (0.313)
Schema  1: ! (0.460) | . (0.455) |  again (0.401) |  from (0.360) |  but (0.341) |  in (0.321) | ." (0.307) |  when (0.305)
Schema  2: ! (0.427) | . (0.416) |  again (0.373) |  from (0.344) |  and (0.329) |  but (0.314) |  too (0.304) |  in (0.304)
Schema  3: . (0.398) |  and (0.384) |  the (0.367) |  in (0.341) |  smiled (0.329) |  happy (0.306) |  sad (0.305) |  again (0.302)
Schema  4: . (0.441) | ! (0.440) |  in (0.417) |  when (0.415) |  again (0.415) |  on (0.408) |  from (0.400) |  the (0.396)
Schema  5: " (0.352) |  says (0.299) |  smiled (0.267) | ! (0.261) | . (0.252) |  the (0.249) |  replied (0.244) |  again (0.241)
Schema  6: " (0.532) |  from (0.325) |  replied (0.321) |  again (0.321) | ! (0.312) | M (0.297) |  water (0.296) |  happened (0.293)
```

### **Generative Behavior**

* **Chemistry task:** generates syntactically valid SMILES-like sequences with plausible structural recurrence.
* **Language task:** produces rhythmic but semantically inconsistent text; indicative of partial coherence and under-trained schema adaptation.

### **Compute and Domain Decision**

> Due to limited computational resources, large-scale NLP evaluation was not feasible.
> Preliminary results were obtained only on small text corpora.
> Consequently, subsequent experiments focused on **chemical language modeling**, where smaller vocabularies and well-defined compositional rules allowed more efficient investigation of schema abstraction and memory dynamics.

### **Interim Assessment**

> SCMT exhibits interpretable schema formation, stable entropy dynamics, and consistent memory utilization.
> Further scaling is required to test its full language modeling potential, but results in molecular sequence modeling validate its underlying mechanisms.

---

## **Current Status**

| Aspect                  | Status                     |
| ----------------------- | -------------------------- |
| Training Stability      | ✅ Stable                   |
| Schema Interpretability | ✅ Emerging                 |
| Cross-domain Behavior   | ⚙️ Preliminary             |
| NLP Evaluation          | ⚠️ Limited (compute-bound) |
| Chemical Modeling       | ✅ Active focus             |
| Benchmarks              | 🚧 In Progress             |
| Safety & Fairness       | ❌ Not Assessed             |

---

## **Citation**
```bibtex
@software{ssru2025scmt,
  author       = {G,, Q., and V.},
  title        = {Schema-Augmented Self-Optimizing Memory Transformer (SCMT)},
  year         = {2025},
  organization = {Sublation Systems Research Unit (SSRU)},
  note         = {Experimental Research Prototype},
  url          = {https://github.com/ssru1010/SCMT},
  version      = {0.1.0},
}
```
