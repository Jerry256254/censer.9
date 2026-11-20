from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 192
    d_ff: int = 512
    max_seq_len: int = 192
    dropout: float = 0.1
    tie_embeddings: bool = True
    pad_id: int = 0
    answer_ids: tuple[int, ...] = (0, 1)
    # HF-compatible linear RoPE scaling: extends context to max_seq_len without
    # retraining the frequencies (factor = max_seq_len / training_length).
    rope_scaling: dict | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0, rope_scaling: dict | None = None):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    # HF "linear" rope scaling: inv_freq /= factor (== positions / factor)
    if rope_scaling and (rope_scaling.get("type") == "linear" or rope_scaling.get("rope_type") == "linear"):
        factor = float(rope_scaling.get("factor", 1.0))
        if factor > 1.0:
            freqs = freqs / factor
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    cos = torch.cat((cos, cos), dim=-1)[:, None, :]
    sin = torch.cat((sin, sin), dim=-1)[:, None, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim)
        k = k.view(b, t, self.n_heads, self.head_dim)
        v = v.view(b, t, self.n_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class KucLabCneser(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("answer_id_tensor", torch.tensor(cfg.answer_ids, dtype=torch.long), persistent=False)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Separate classification head (NOT tied to embedding rows): tying made the
        # category logits collide with the real tokens "a".."j" in normal text.
        self.score = nn.Linear(cfg.d_model, len(cfg.answer_ids), bias=False)
        self.apply(self._init)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._rope_cache: tuple[torch.Tensor, torch.Tensor, str] | None = None

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def _rope(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = f"{device}|{dtype}"
        if self._rope_cache is None or self._rope_cache[2] != key:
            head_dim = self.cfg.d_model // self.cfg.n_heads
            cos, sin = build_rope_cache(self.cfg.max_seq_len, head_dim, device, dtype, rope_scaling=self.cfg.rope_scaling)
            self._rope_cache = (cos, sin, key)
        return self._rope_cache[0], self._rope_cache[1]

    def _attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        pad = input_ids != self.cfg.pad_id
        causal = torch.ones(t, t, dtype=torch.bool, device=input_ids.device).tril()
        mask = pad[:, None, None, :] & causal[None, None, :, :]
        return mask

    def _hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        cos, sin = self._rope(input_ids.device, self.embed.weight.dtype)
        mask = self._attention_mask(input_ids)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.norm(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self._hidden(input_ids))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def last_token_index(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Index of the final non-pad token per row."""
        non_pad = input_ids != self.cfg.pad_id
        lengths = non_pad.sum(dim=1).clamp(min=1)
        return lengths - 1

    def answer_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits over the answer head: [B, len(answer_ids)] (2 for binary, 12 for multi-label)."""
        hidden = self._hidden(input_ids)
        logits = hidden.gather(1, self.last_token_index(input_ids)[:, None, None].expand(-1, 1, hidden.shape[-1]))
        return self.score(logits[:, 0, :].float())

    @torch.no_grad()
    def classify(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Multi-label: sigmoid probabilities over all answer tokens [B, n_answers]."""
        return self.answer_logits(input_ids).sigmoid()

    @torch.no_grad()
    def classify_binary(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Binary unsafe probability (answer token "1") — comparable with .1-.5."""
        return self.classify(input_ids)[:, 1]


def estimate_params(cfg: ModelConfig) -> int:
    emb = cfg.vocab_size * cfg.d_model
    per_layer = 4 * cfg.d_model * cfg.d_model + 3 * cfg.d_model * cfg.d_ff + 2 * cfg.d_model
    return emb + cfg.n_layers * per_layer + cfg.d_model


def remap_legacy_state(state: dict) -> dict:
    """Old checkpoints tied the answer head to embedding rows; materialize score.weight."""
    model = state["model"]
    if "score.weight" not in model:
        ans = list(state.get("model_config", {}).get("answer_ids", (0, 1)))
        model["score.weight"] = model["embed.weight"][ans].clone().contiguous()
    return state
