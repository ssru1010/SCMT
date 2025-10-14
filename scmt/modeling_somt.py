"""SchemaAugmentedSOMT modular model file.
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
    Saves config.json and state dict.
    """
    config_class = SOMTConfig

    @classmethod
    def from_pretrained(cls, load_dir: str, map_location: Optional[str] = None):
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
        os.makedirs(save_dir, exist_ok=True)
        cfg = self.get_config().to_dict()
        _save_config(cfg, save_dir)
        state_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), state_path)

    def get_config(self) -> SOMTConfig:
        raise NotImplementedError


class SchemaAugmentedSOMT(SOMTPreTrainedModel):
    def __init__(self, vocab_size: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 max_len: int = 256, mem_size: int = 128,
                 num_schemas: int = 32,
                 recency_lambda: float = 0.01,
                 entropy_reg_coef: float = 1e-3,
                 l2_importance_coef: float = 1e-4,
                 schema_utility_coef: float = 1e-2):
        super().__init__()
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

        self.query_proj = nn.Linear(d_model, d_model)
        self.uncertainty_gate = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, d_model), nn.Sigmoid())

        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.retrieval_fuse = nn.Linear(d_model, d_model)

        self.threshold_net = nn.Sequential(nn.Linear(d_model + 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.budget_controller = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        self.importance_net = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.obsolescence_net = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        # persistent schema buffers
        self.register_buffer("schema_keys", torch.randn(num_schemas, d_model) * 0.02)
        self.register_buffer("schema_vals", torch.randn(num_schemas, d_model) * 0.02)
        self.register_buffer("schema_utility_sum", torch.zeros(num_schemas))
        self.register_buffer("schema_usage_count", torch.zeros(num_schemas))

        self.schema_router = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_schemas))
        self.schema_utility_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # tie weights
        try:
            self.lm_head.weight = self.embed.weight
        except Exception:
            pass

        self.mem_dropout = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

    def _form_schemas(self, mem_keys, mem_vals):
        # mem_keys: (B, M, D)
        B, M, D = mem_keys.shape
        if M == 0:
            return self.schema_keys.unsqueeze(0).expand(B, -1, -1), torch.zeros(B, M, self.num_schemas, device=mem_keys.device)
        routing_logits = self.schema_router(mem_keys)
        routing_probs = F.softmax(routing_logits, dim=-1)
        schema_reps = torch.einsum('bms,bmd->bsd', routing_probs, mem_keys)
        return schema_reps, routing_probs

    def forward(self, x: torch.Tensor, memory_keys: Optional[torch.Tensor] = None, memory_vals: Optional[torch.Tensor] = None, memory_age: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward returns (logits, updated_keys, updated_vals, updated_age, aux)

        x: (B, L) token ids
        memory_*: optional tensors representing previous memory states.
        """
        B, L = x.shape
        device = x.device

        x_emb = self.embed(x) + self.pos_embed[:, :L]
        encoded = self.encoder(x_emb, mask=self._causal_mask(L, device))

        base_logits = self.lm_head(encoded)
        probs = F.softmax(base_logits.detach(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        e_min = entropy.min(dim=1, keepdim=True).values
        e_max = entropy.max(dim=1, keepdim=True).values
        entropy_norm = (entropy - e_min) / (e_max - e_min + 1e-12)

        uncertainty_factor = self.uncertainty_gate(encoded)
        queries = self.query_proj(encoded) * (1 + uncertainty_factor)

        avg_ent = entropy_norm.mean(dim=1, keepdim=True)
        max_ent = entropy_norm.max(dim=1, keepdim=True).values
        thresh_input = torch.cat([encoded, avg_ent.unsqueeze(1).expand(-1, L, -1), max_ent.unsqueeze(1).expand(-1, L, -1)], dim=-1)
        dynamic_threshold = self.threshold_net(thresh_input).squeeze(-1)

        batch_context = encoded.mean(dim=1)
        budget_frac = self.budget_controller(batch_context).squeeze(-1)

        if memory_keys is None:
            memory_keys = torch.empty(B, 0, self.d_model, device=device)
            memory_vals = torch.empty(B, 0, self.d_model, device=device)
            memory_age = torch.empty(B, 0, device=device)

        updated_keys_list, updated_vals_list, updated_age_list = [], [], []
        entropy_reg_total = torch.tensor(0.0, device=device)
        l2_importance_total = torch.tensor(0.0, device=device)

        for b in range(B):
            mask = entropy_norm[b] > dynamic_threshold[b]
            if mask.any():
                candidate_k = self.key_proj(encoded[b][mask])
                candidate_v = self.value_proj(x_emb[b][mask])
                candidate_age = torch.zeros(candidate_k.size(0), device=device)
            else:
                candidate_k = torch.empty(0, self.d_model, device=device)
                candidate_v = torch.empty(0, self.d_model, device=device)
                candidate_age = torch.empty(0, device=device)

            old_k = memory_keys[b] if memory_keys.size(1) > 0 else torch.empty(0, self.d_model, device=device)
            old_v = memory_vals[b] if memory_vals.size(1) > 0 else torch.empty(0, self.d_model, device=device)
            old_age = (memory_age[b] + 1) if memory_age.size(1) > 0 else torch.empty(0, device=device)

            if old_k.numel() == 0:
                combined_k = candidate_k
                combined_v = candidate_v
                combined_age = candidate_age
            else:
                combined_k = torch.cat([old_k, candidate_k], dim=0)
                combined_v = torch.cat([old_v, candidate_v], dim=0)
                combined_age = torch.cat([old_age, candidate_age], dim=0)

            n_total = combined_k.size(0)
            if n_total > self.mem_size:
                importance = self.importance_net(combined_k).squeeze(-1)

                if old_k.size(0) > 0:
                    obsolescence_scores = self.obsolescence_net(old_k).squeeze(-1)
                    age_penalty = old_age.float() * obsolescence_scores
                    full_penalty = torch.cat([age_penalty, torch.zeros(candidate_k.size(0), device=device)], dim=0)
                    importance = importance - full_penalty

                if candidate_k.size(0) > 0:
                    recency_bonus = torch.zeros_like(importance)
                    new_idx = torch.arange(n_total - candidate_k.size(0), n_total, device=device)
                    recency_bonus[new_idx] = torch.linspace(1.0, 0.1, steps=candidate_k.size(0), device=device)
                    importance += self.recency_lambda * recency_bonus

                probs_imp = F.softmax(importance, dim=0)
                entropy_reg_total += (-torch.sum(probs_imp * torch.log(probs_imp + 1e-12)))
                l2_importance_total += torch.mean(importance ** 2)

                _, topk_idx = torch.topk(importance, k=self.mem_size, largest=True)
                combined_k = combined_k[topk_idx]
                combined_v = combined_v[topk_idx]
                combined_age = combined_age[topk_idx]

            pad_len = self.mem_size - combined_k.size(0)
            if pad_len > 0:
                pad_k = torch.vstack([combined_k, torch.zeros(pad_len, self.d_model, device=device)])
                pad_v = torch.vstack([combined_v, torch.zeros(pad_len, self.d_model, device=device)])
                pad_age = torch.cat([combined_age, torch.zeros(pad_len, device=device)])
            else:
                pad_k = combined_k
                pad_v = combined_v
                pad_age = combined_age

            updated_keys_list.append(pad_k)
            updated_vals_list.append(pad_v)
            updated_age_list.append(pad_age)

        updated_keys = torch.stack(updated_keys_list, dim=0)
        updated_vals = torch.stack(updated_vals_list, dim=0)
        updated_age = torch.stack(updated_age_list, dim=0)

        schema_utility_loss = torch.tensor(0.0, device=device)
        final_context = torch.zeros_like(encoded)

        if self.training and updated_keys.size(1) > 0:
            batch_schema_keys, routing_probs = self._form_schemas(updated_keys, updated_vals)
            with torch.no_grad():
                batch_mean = batch_schema_keys.mean(dim=0)
                # EMA update to persistent schema keys
                self.schema_keys = 0.99 * self.schema_keys + 0.01 * batch_mean
                self.schema_vals = self.schema_keys.clone()

            current_schema_keys = self.schema_keys.unsqueeze(0).expand(B, -1, -1)
            schema_scores = torch.matmul(queries, current_schema_keys.transpose(-2, -1)) / math.sqrt(self.d_model)
            schema_weights = F.softmax(schema_scores, dim=-1)
            schema_context = torch.matmul(schema_weights, current_schema_keys)

            instance_scores = torch.bmm(queries, updated_keys.transpose(1, 2)) / math.sqrt(self.d_model)
            instance_weights = F.softmax(instance_scores, dim=-1)
            instance_context = torch.bmm(instance_weights, updated_vals)

            fused_context = schema_context + instance_context
            final_context = self.retrieval_fuse(self.mem_dropout(fused_context))

            with torch.no_grad():
                enhanced_logits = self.lm_head(encoded + final_context)
                enhanced_probs = F.softmax(enhanced_logits, dim=-1)
                enhanced_entropy = -torch.sum(enhanced_probs * torch.log(enhanced_probs + 1e-12), dim=-1)
                entropy_reduction = entropy_norm - enhanced_entropy
                target_utility = entropy_reduction.mean().clamp(min=0)

            pred_utility = self.schema_utility_head(current_schema_keys.mean(dim=1)).mean()
            schema_utility_loss = (pred_utility - target_utility) ** 2

        elif updated_keys.size(1) > 0:
            current_schema_keys = self.schema_keys.unsqueeze(0).expand(B, -1, -1)
            schema_scores = torch.matmul(queries, current_schema_keys.transpose(-2, -1)) / math.sqrt(self.d_model)
            schema_weights = F.softmax(schema_scores, dim=-1)
            schema_context = torch.matmul(schema_weights, current_schema_keys)

            instance_scores = torch.bmm(queries, updated_keys.transpose(1, 2)) / math.sqrt(self.d_model)
            instance_weights = F.softmax(instance_scores, dim=-1)
            instance_context = torch.bmm(instance_weights, updated_vals)

            fused_context = schema_context + instance_context
            final_context = self.retrieval_fuse(self.mem_dropout(fused_context))

        encoded = encoded + final_context
        logits = self.lm_head(encoded)

        aux = {
            "importance_entropy": entropy_reg_total * self.entropy_reg_coef,
            "importance_l2": l2_importance_total * self.l2_importance_coef,
            "schema_utility": schema_utility_loss * self.schema_utility_coef,
        }

        return logits, updated_keys, updated_vals, updated_age, aux

    # Wrapper expected by HF users
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

        # Initialize input
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

        for _ in range(max_length):
            logits, mem_k, mem_v, mem_age, _ = self(input_ids, mem_k, mem_v, mem_age)
            next_logits = logits[:, -1, :]

            # Repetition penalty
            for token_id in set(generated):
                if 0 <= token_id < next_logits.size(-1):
                    next_logits[:, token_id] /= repetition_penalty

            # Temperature + sampling
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
