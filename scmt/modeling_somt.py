"""
Schema-Augmented Self-Optimizing Memory Transformer (SCMT)

Experimental research implementation of an adaptive transformer
architecture integrating entropy-regulated memory allocation and
schema-level abstraction. Designed for interpretability, modularity,
and extensibility.

G,, Q., & V. (2025). Schema-Augmented Self-Optimizing Memory Transformer (SCMT).
Sublation Systems Research Unit (SSRU). Experimental Research Prototype.
Version 0.1.1 — https://github.com/ssru1010/SCMT  

SchemaAugmentedSOMT modular model file.
Contains:
 - SOMTConfig dataclass
 - SOMTPreTrainedModel base: save/load helpers
 - SchemaAugmentedSOMT: main nn.Module

This file is intentionally self-contained and focuses on API compatibility
(similar to HuggingFace style) for loading/saving and generation.
"""

from dataclasses import dataclass, asdict
import json
import os
from typing import Optional, Tuple, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_utils import _save_config, _load_config, top_k_top_p_filtering


@dataclass
class SOMTConfig:
    """Configuration dataclass for SchemaAugmentedSOMT."""
    vocab_size: int = 50257
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    max_len: int = 256
    mem_size: int = 128
    num_schemas: int = 32
    recency_lambda: float = 0.01
    entropy_reg_coef: float = 1e-3
    l2_importance_coef: float = 1e-4
    schema_utility_coef: float = 1e-2

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SOMTPreTrainedModel(nn.Module):
    """Minimal PreTrainedModel-like helper for save/load.
    Provides save_pretrained() / from_pretrained() API compatibility.
    """
    config_class = SOMTConfig

    @classmethod
    def from_pretrained(cls, load_dir: str, map_location: Optional[str] = None):
        """Load a pretrained model from a directory."""
        cfg = SOMTConfig.from_dict(_load_config(load_dir))
        model = cls(cfg.vocab_size, **{k: v for k, v in cfg.to_dict().items() if k != "vocab_size"})
        state_path = os.path.join(load_dir, "pytorch_model.bin")
        if os.path.exists(state_path):
            sd = torch.load(state_path, map_location=map_location)
            model.load_state_dict(sd)
        else:
            raise FileNotFoundError(f"pytorch_model.bin not found in {load_dir}")
        return model

    def save_pretrained(self, save_dir: str):
        """Save model weights and configuration."""
        os.makedirs(save_dir, exist_ok=True)
        cfg = self.get_config().to_dict()
        _save_config(cfg, save_dir)
        state_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), state_path)

    def get_config(self) -> SOMTConfig:
        return self.config



class SchemaAugmentedSOMT(SOMTPreTrainedModel):
    """Schema-Augmented Self-Optimizing Memory Transformer (SCMT/SOMT).

    Extends a causal Transformer encoder with:
    • Adaptive episodic memory (importance-weighted, recency-decayed)
    • Schema abstraction layer (persistent schema buffers)
    • Entropy-regulated uncertainty gating and memory allocation
    """
    def __init__(self, vocab_size: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 max_len: int = 256, mem_size: int = 128,
                 num_schemas: int = 32,
                 recency_lambda: float = 0.01,
                 entropy_reg_coef: float = 1e-3,
                 l2_importance_coef: float = 1e-4,
                 schema_utility_coef: float = 1e-2):
        super().__init__()

        # Core configuration
        self.config = SOMTConfig(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers,
            max_len=max_len, mem_size=mem_size, num_schemas=num_schemas,
            recency_lambda=recency_lambda, entropy_reg_coef=entropy_reg_coef,
            l2_importance_coef=l2_importance_coef, schema_utility_coef=schema_utility_coef
        )
        self.d_model = d_model
        self.max_len = max_len
        self.mem_size = mem_size
        self.num_schemas = num_schemas
        self.recency_lambda = recency_lambda
        self.entropy_reg_coef = entropy_reg_coef
        self.l2_importance_coef = l2_importance_coef
        self.schema_utility_coef = schema_utility_coef

        # embeddings + positional
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        
        # Uncertainty-gated query projection for entropy modulation
        self.query_proj = nn.Linear(d_model, d_model)
        self.uncertainty_gate = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, d_model), nn.Sigmoid())

        # Episodic memory write/read projections
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.retrieval_fuse = nn.Linear(d_model, d_model)

        # Dynamic write threshold and global budget controllers
        self.threshold_net = nn.Sequential(nn.Linear(d_model + 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.budget_controller = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        # Importance and obsolescence predictors for pruning
        self.importance_net = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.obsolescence_net = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        # Persistent schema buffers → trainable parameters (emergent but fixed post-training)
        # These act as global abstraction units that self-organize during training.
        self.schema_keys = nn.Parameter(torch.randn(num_schemas, d_model) * 0.02)
        self.schema_vals = nn.Parameter(torch.randn(num_schemas, d_model) * 0.02)

        # Track per-schema utility statistics as non-trainable buffers
        self.register_buffer("schema_utility_sum", torch.zeros(num_schemas))
        self.register_buffer("schema_usage_count", torch.zeros(num_schemas))

        # Schema router and utility head
        self.schema_router = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_schemas))
        self.schema_utility_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

        # Language modeling head (weight-tied)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        try:
            self.lm_head.weight = self.embed.weight
        except Exception:
            pass

        self.mem_dropout = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Xavier init for linear layers, normal init for embeddings."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, L, device):
        """Create additive causal mask: -inf for future positions, 0 for allowed."""
        mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor,
                memory_keys: Optional[torch.Tensor] = None,
                memory_vals: Optional[torch.Tensor] = None,
                memory_age: Optional[torch.Tensor] = None,
                precomputed_entropy_norm: Optional[torch.Tensor] = None,
                precomputed_dynamic_threshold: Optional[torch.Tensor] = None,
                step_mode: bool = False,
                global_pos_offset: int = 0,  # PATCH: added for causal continuity
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        B, L = x.shape
        device = x.device

        # ---- Explicit causal mask for encoder (critical fix) ----
        causal_attn_mask = self._causal_mask(L, device)

        # ---- Encoding with explicit mask ----
        x_emb = self.embed(x) + self.pos_embed[:, :L]
        encoded = self.encoder(x_emb, mask=causal_attn_mask)

        # ---- Compute token-level entropy as uncertainty measure ----
        base_logits = self.lm_head(encoded)
        probs = F.softmax(base_logits.detach(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

        if precomputed_entropy_norm is None:
            e_mean = entropy.mean(dim=1, keepdim=True)
            e_std = entropy.std(dim=1, keepdim=True, unbiased=False) + 1e-12
            entropy_norm = torch.sigmoid((entropy - e_mean) / e_std)
        else:
            entropy_norm = precomputed_entropy_norm.to(device)

        # ---- Uncertainty gating and thresholding ----
        uncertainty_factor = self.uncertainty_gate(encoded)
        queries = self.query_proj(encoded) * (1 + uncertainty_factor)

        cumulative_sum = torch.cumsum(entropy_norm, dim=1)
        steps = torch.arange(1, L + 1, device=device).float().unsqueeze(0)
        causal_mean = cumulative_sum / steps
        causal_max, _ = torch.cummax(entropy_norm, dim=1)

        thresh_input = torch.cat(
            [encoded, causal_mean.unsqueeze(-1), causal_max.unsqueeze(-1)], dim=-1
        )
        dynamic_threshold = self.threshold_net(thresh_input).squeeze(-1)

        if precomputed_dynamic_threshold is not None:
            dynamic_threshold = precomputed_dynamic_threshold.to(device)

        batch_context = encoded.mean(dim=1)
        budget_frac = self.budget_controller(batch_context).squeeze(-1)

        # ---- Episodic memory write/prune (causally safe) ----
        if memory_keys is None:
            memory_keys = torch.empty(B, 0, self.d_model, device=device)
            memory_vals = torch.empty(B, 0, self.d_model, device=device)
            memory_age = torch.empty(B, 0, device=device)

        updated_keys_list, updated_vals_list, updated_age_list = [], [], []
        updated_times_list = []
        # PATCH: initialize aux accumulators as scalars on device
        entropy_reg_total = torch.tensor(0.0, device=device)
        l2_importance_total = torch.tensor(0.0, device=device)

        for b in range(B):
            old_k = memory_keys[b] if memory_keys.size(1) > 0 else torch.empty(0, self.d_model, device=device)
            old_v = memory_vals[b] if memory_vals.size(1) > 0 else torch.empty(0, self.d_model, device=device)
            old_age = memory_age[b] if memory_age.size(1) > 0 else torch.empty(0, device=device)

            mem_k = old_k.clone() if old_k.numel() else torch.empty(0, self.d_model, device=device)
            mem_v = old_v.clone() if old_v.numel() else torch.empty(0, self.d_model, device=device)
            mem_a = old_age.clone() if old_age.numel() else torch.empty(0, device=device)

            if old_age.numel():
                mem_times = old_age.clone().long().to(device)  # reuse age as timestamp storage
            else:
                mem_times = torch.empty(0, dtype=torch.long, device=device)

            time_steps = [L - 1] if step_mode else range(L)
            for t in time_steps:
                ent_t = entropy_norm[b, t]
                thr_t = dynamic_threshold[b, t]
                if ent_t > thr_t:
                    cand_k = self.key_proj(encoded[b, t:t+1])
                    cand_v = self.value_proj(x_emb[b, t:t+1])
                    cand_a = torch.zeros(1, device=device)

                    # PATCH: accumulate aux losses on write
                    entropy_reg_total = entropy_reg_total + ent_t
                    l2_importance_total = l2_importance_total + (cand_k.norm(p=2) + cand_v.norm(p=2)) * 0.5

                    mem_k = torch.cat([mem_k, cand_k], dim=0) if mem_k.numel() else cand_k.clone()
                    mem_v = torch.cat([mem_v, cand_v], dim=0) if mem_v.numel() else cand_v.clone()
                    mem_a = torch.cat([mem_a, cand_a], dim=0) if mem_a.numel() else cand_a.clone()
                    # PATCH: use global_pos_offset for absolute timestamp
                    t_tensor = torch.full((1,), t + int(global_pos_offset), dtype=torch.long, device=device)
                    mem_times = torch.cat([mem_times, t_tensor], dim=0) if mem_times.numel() else t_tensor.clone()

                if mem_a.numel() > 0:
                    mem_a = mem_a + 1

                if mem_k.size(0) > self.mem_size:
                    importance = self.importance_net(mem_k).squeeze(-1)
                    # PATCH: accumulate importance stats
                    l2_importance_total = l2_importance_total + importance.pow(2).mean()

                    if mem_times.numel() > 0:
                        age_slots = (t + global_pos_offset - mem_times).float().clamp(min=0.0)
                        obsolescence_scores = self.obsolescence_net(mem_k).squeeze(-1)
                        age_penalty = age_slots * obsolescence_scores
                        importance = importance - age_penalty
                    topk = min(self.mem_size, importance.size(0))
                    _, top_idx = torch.topk(importance, k=topk, largest=True)
                    mem_k = mem_k[top_idx]
                    mem_v = mem_v[top_idx]
                    mem_a = mem_a[top_idx]
                    mem_times = mem_times[top_idx]

            pad_len = self.mem_size - mem_k.size(0)
            if pad_len > 0:
                pad_k = torch.cat([mem_k, torch.zeros(pad_len, self.d_model, device=device)], dim=0) if mem_k.numel() else torch.zeros(self.mem_size, self.d_model, device=device)
                pad_v = torch.cat([mem_v, torch.zeros(pad_len, self.d_model, device=device)], dim=0) if mem_v.numel() else torch.zeros(self.mem_size, self.d_model, device=device)
                pad_a = torch.cat([mem_a, torch.zeros(pad_len, device=device)], dim=0) if mem_a.numel() else torch.zeros(self.mem_size, device=device)
                future_pad = torch.full((pad_len,), 10**9, dtype=torch.long, device=device)
                pad_times = torch.cat([mem_times, future_pad], dim=0) if mem_times.numel() else torch.full((self.mem_size,), 10**9, dtype=torch.long, device=device)
            else:
                pad_k, pad_v, pad_a = mem_k, mem_v, mem_a
                pad_times = mem_times

            updated_keys_list.append(pad_k)
            updated_vals_list.append(pad_v)
            updated_age_list.append(pad_a)
            updated_times_list.append(pad_times)

        updated_keys = torch.stack(updated_keys_list, dim=0)
        updated_vals = torch.stack(updated_vals_list, dim=0)
        updated_age = torch.stack(updated_age_list, dim=0)
        updated_times = torch.stack(updated_times_list, dim=0)

        # ---- Causal retrieval & schema abstraction ----
        time_idx = torch.arange(L, device=device).unsqueeze(0).unsqueeze(-1)      # (1, L, 1)
        mem_times_exp = updated_times.unsqueeze(1)                                # (B, 1, M)
        retrieval_block_mask = mem_times_exp > (time_idx + global_pos_offset)     # PATCH: include offset in mask

        instance_scores = torch.bmm(queries, updated_keys.transpose(1, 2)) / math.sqrt(self.d_model)
        instance_scores = instance_scores.masked_fill(retrieval_block_mask, float('-inf'))
        instance_weights = F.softmax(instance_scores, dim=-1)
        instance_weights = torch.nan_to_num(instance_weights, nan=0.0)
        instance_context = torch.bmm(instance_weights, updated_vals)

        current_schema_keys = self.schema_keys.unsqueeze(0).expand(B, -1, -1)
        schema_scores = torch.matmul(queries, current_schema_keys.transpose(-2, -1)) / math.sqrt(self.d_model)
        schema_weights = F.softmax(schema_scores, dim=-1)
        schema_context = torch.matmul(schema_weights, self.schema_vals.unsqueeze(0).expand(B, -1, -1))

        fused_context = self.retrieval_fuse(self.mem_dropout(schema_context + instance_context))
        encoded = encoded + fused_context

        # --- PATCH: per-schema utility loss ---
        current_schema_vals = self.schema_vals.unsqueeze(0).expand(B, -1, -1)
        per_schema_cos = F.cosine_similarity(current_schema_keys, current_schema_vals, dim=-1).clamp(-1.0, 1.0)
        per_schema_alignment_pos = (1.0 - per_schema_cos) / 2.0  # [0,1], high when misaligned
        per_schema_activity = schema_weights.mean(dim=(0, 1))   # (S,)

        schema_utility_loss = (per_schema_alignment_pos.mean(dim=0) * per_schema_activity).sum() / (per_schema_activity.sum().clamp(min=1e-6))

        with torch.no_grad():
            alpha = 0.05
            per_schema_usage = per_schema_activity.detach()
            self.schema_utility_sum = (1 - alpha) * self.schema_utility_sum + alpha * per_schema_usage
            self.schema_usage_count = self.schema_usage_count + per_schema_usage

        logits = self.lm_head(encoded)

        aux = {
            "importance_entropy": entropy_reg_total * self.entropy_reg_coef,
            "importance_l2": l2_importance_total * self.l2_importance_coef,
            "schema_utility": schema_utility_loss * self.schema_utility_coef,
            "entropy_norm": entropy_norm.detach(),
            "dynamic_threshold": dynamic_threshold.detach(),
            "memory_timestamps": updated_times.detach(),
        }

        return logits, updated_keys, updated_vals, updated_age, aux


    @torch.no_grad()
    def generate(
        self,
        tokenizer=None,
        prompt: str = "",
        input_ids: Optional[torch.Tensor] = None,
        max_length: int = 80,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        device = next(self.parameters()).device
        self.eval()

        if input_ids is None:
            if tokenizer is not None and prompt:
                toks = tokenizer.encode(prompt, add_special_tokens=False)
                if not toks:
                    toks = [tokenizer.bos_token_id or 0]
                input_ids = torch.tensor([toks], dtype=torch.long, device=device)
            else:
                start_id = eos_token_id or pad_token_id or 0
                input_ids = torch.tensor([[start_id]], dtype=torch.long, device=device)

        generated = input_ids[0].tolist()
        mem_k, mem_v, mem_age = None, None, None
        # PATCH: track global position for causal continuity
        global_t = input_ids.size(1) if input_ids is not None else 0

        for _ in range(max_length):
            # PATCH: use step_mode=True and global_pos_offset
            logits, mem_k, mem_v, mem_age, _ = self(
                input_ids, mem_k, mem_v, mem_age,
                step_mode=True,
                global_pos_offset=global_t
            )
            next_logits = logits[:, -1, :]

            for token_id in set(generated):
                if 0 <= token_id < next_logits.size(-1):
                    next_logits[:, token_id] /= repetition_penalty

            next_logits = next_logits / max(temperature, 1e-6)
            next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)

            if not torch.isfinite(next_logits).any():
                next_logits = torch.zeros_like(next_logits)

            probs = F.softmax(next_logits, dim=-1)
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())

            if token_id in {eos_token_id, pad_token_id}:
                break

            generated.append(token_id)
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
            global_t += 1  # PATCH: increment global timestep

            if len(generated) >= self.max_len:
                break

        return (
            tokenizer.decode(generated, skip_special_tokens=True)
            if tokenizer is not None
            else generated
        )


# convenience alias for importers
SOMTConfig = SOMTConfig
SchemaAugmentedSOMT = SchemaAugmentedSOMT