"""
Schema-Augmented Self-Optimizing Memory Transformer (SCMT)

Experimental research implementation of an adaptive transformer
architecture integrating entropy-regulated memory allocation and
schema-level abstraction. Designed for interpretability, modularity,
and extensibility.

G,, Q., & V. (2025). Schema-Augmented Self-Optimizing Memory Transformer (SCMT).
Sublation Systems Research Unit (SSRU). Experimental Research Prototype.
Version 0.1.2 — Memory Leak Patched

SchemaAugmentedSOMT modular model file.
Contains:
 - SOMTConfig dataclass
 - SOMTPreTrainedModel base: save/load helpers
 - SchemaAugmentedSOMT: main nn.Module

MEMORY LEAK FIXES:
- Detached auxiliary loss accumulation
- Pre-allocated tensor buffers for memory operations
- Explicit gradient blocking in pruning loops
- Cleaned up timestamp tracking
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
                global_pos_offset: int = 0,
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

        # FIX: Pre-allocate output buffers to avoid repeated concatenation
        updated_keys = torch.zeros(B, self.mem_size, self.d_model, device=device)
        updated_vals = torch.zeros(B, self.mem_size, self.d_model, device=device)
        updated_age = torch.zeros(B, self.mem_size, device=device)
        updated_times = torch.full((B, self.mem_size), 10**9, dtype=torch.long, device=device)
        
        # FIX: Detached auxiliary loss accumulators (no gradient retention)
        entropy_reg_sum = 0.0
        l2_importance_sum = 0.0
        total_writes = 0

        for b in range(B):
            # FIX: Work with detached copies for metadata
            old_k = memory_keys[b] if memory_keys.size(1) > 0 else torch.empty(0, self.d_model, device=device)
            old_v = memory_vals[b] if memory_vals.size(1) > 0 else torch.empty(0, self.d_model, device=device)
            old_age = memory_age[b] if memory_age.size(1) > 0 else torch.empty(0, device=device)

            # Start with existing memory
            current_size = min(old_k.size(0), self.mem_size)
            mem_k = old_k[:current_size].clone()
            mem_v = old_v[:current_size].clone()
            mem_a = old_age[:current_size].clone()
            
            # FIX: Detach timestamps from computational graph
            if old_age.numel():
                mem_times = old_age[:current_size].clone().detach().long()
            else:
                mem_times = torch.empty(0, dtype=torch.long, device=device)

            time_steps = [L - 1] if step_mode else range(L)
            
            for t in time_steps:
                # FIX: Detach entropy for auxiliary loss (no gradient needed)
                ent_t = entropy_norm[b, t].item()  # Convert to Python float
                thr_t = dynamic_threshold[b, t].item()
                
                if ent_t > thr_t:
                    cand_k = self.key_proj(encoded[b, t:t+1])
                    cand_v = self.value_proj(x_emb[b, t:t+1])
                    cand_a = torch.zeros(1, device=device)

                    # FIX: Accumulate as Python floats (no graph retention)
                    with torch.no_grad():
                        entropy_reg_sum += ent_t
                        l2_importance_sum += (cand_k.norm(p=2).item() + cand_v.norm(p=2).item()) * 0.5
                    total_writes += 1

                    # Append new memory
                    mem_k = torch.cat([mem_k, cand_k], dim=0)
                    mem_v = torch.cat([mem_v, cand_v], dim=0)
                    mem_a = torch.cat([mem_a, cand_a], dim=0)
                    t_tensor = torch.tensor([t + global_pos_offset], dtype=torch.long, device=device)
                    mem_times = torch.cat([mem_times, t_tensor], dim=0)

                # Age existing memory
                if mem_a.numel() > 0:
                    mem_a = mem_a + 1

                # Prune if over capacity
                if mem_k.size(0) > self.mem_size:
                    # FIX: Detach importance computation from gradient flow
                    with torch.no_grad():
                        importance = self.importance_net(mem_k.detach()).squeeze(-1)
                        
                        if mem_times.numel() > 0:
                            age_slots = (t + global_pos_offset - mem_times).float().clamp(min=0.0)
                            obsolescence_scores = self.obsolescence_net(mem_k.detach()).squeeze(-1)
                            age_penalty = age_slots * obsolescence_scores
                            importance = importance - age_penalty
                        
                        topk = min(self.mem_size, importance.size(0))
                        _, top_idx = torch.topk(importance, k=topk, largest=True)
                    
                    # Apply pruning with gradient flow intact
                    mem_k = mem_k[top_idx]
                    mem_v = mem_v[top_idx]
                    mem_a = mem_a[top_idx]
                    mem_times = mem_times[top_idx]

            # FIX: Direct assignment to pre-allocated buffer
            actual_size = min(mem_k.size(0), self.mem_size)
            updated_keys[b, :actual_size] = mem_k[:actual_size]
            updated_vals[b, :actual_size] = mem_v[:actual_size]
            updated_age[b, :actual_size] = mem_a[:actual_size]
            updated_times[b, :actual_size] = mem_times[:actual_size]

        # ---- Causal retrieval & schema abstraction ----
        time_idx = torch.arange(L, device=device).unsqueeze(0).unsqueeze(-1)
        mem_times_exp = updated_times.unsqueeze(1)
        retrieval_block_mask = mem_times_exp > (time_idx + global_pos_offset)

        instance_scores = torch.bmm(queries, updated_keys.transpose(1, 2)) / math.sqrt(self.d_model)
        instance_scores = instance_scores.masked_fill(retrieval_block_mask, -1e4)
        instance_weights = F.softmax(instance_scores, dim=-1)
        instance_weights = torch.nan_to_num(instance_weights, nan=0.0)

        instance_context = torch.bmm(instance_weights, updated_vals)

        current_schema_keys = self.schema_keys.unsqueeze(0).expand(B, -1, -1)
        schema_scores = torch.matmul(queries, current_schema_keys.transpose(-2, -1)) / math.sqrt(self.d_model)
        schema_weights = F.softmax(schema_scores, dim=-1)
        schema_context = torch.matmul(schema_weights, self.schema_vals.unsqueeze(0).expand(B, -1, -1))

        fused_context = self.retrieval_fuse(self.mem_dropout(schema_context + instance_context))
        encoded = encoded + fused_context

        # --- Schema utility loss ---
        current_schema_vals = self.schema_vals.unsqueeze(0).expand(B, -1, -1)
        per_schema_cos = F.cosine_similarity(current_schema_keys, current_schema_vals, dim=-1).clamp(-1.0, 1.0)
        per_schema_alignment_pos = (1.0 - per_schema_cos) / 2.0
        per_schema_activity = schema_weights.mean(dim=(0, 1))

        schema_utility_loss = (per_schema_alignment_pos.mean(dim=0) * per_schema_activity).sum() / (per_schema_activity.sum().clamp(min=1e-6))

        with torch.no_grad():
            alpha = 0.05
            per_schema_usage = per_schema_activity.detach()
            self.schema_utility_sum = (1 - alpha) * self.schema_utility_sum + alpha * per_schema_usage
            self.schema_usage_count = self.schema_usage_count + per_schema_usage

        logits = self.lm_head(encoded)

        # FIX: Convert accumulated floats to tensors only at the end
        aux = {
            "importance_entropy": torch.tensor(entropy_reg_sum, device=device) * self.entropy_reg_coef,
            "importance_l2": torch.tensor(l2_importance_sum, device=device) * self.l2_importance_coef,
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
        global_t = input_ids.size(1) if input_ids is not None else 0

        for _ in range(max_length):
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
            global_t += 1

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