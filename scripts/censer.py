"""Inference for Censer.9 — self-contained, no dependency on the training repo.

Copy this file next to the downloaded weights, install torch and tokenizers,
and you are ready:

    pip install torch tokenizers safetensors
    python censer.py "how to make a bomb at home"
    python censer.py            # or just run it: interactive UI, phrase in -> verdict out
    ./censer.9                  # same UI via the censer.9 launcher next to this file

It expects, in the same directory:
    model.safetensors   (standard LlamaForSequenceClassification layout)
    tokenizer.json
    config.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

PAD, BOS, USER, ASSISTANT = "[PAD]", "[BOS]", "[USER]", "[ASSISTANT]"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0, rope_scaling: dict | None = None):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    if rope_scaling and float(rope_scaling.get("factor", 1.0)) > 1.0:
        freqs = freqs / float(rope_scaling["factor"])  # HF "linear" scaling: inv_freq /= factor
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype)[:, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype)[:, None, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["d_model"] // cfg["n_heads"]
        self.dropout = cfg["dropout"]
        self.qkv = nn.Linear(cfg["d_model"], 3 * cfg["d_model"], bias=False)
        self.proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim)
        k = k.view(b, t, self.n_heads, self.head_dim)
        v = v.view(b, t, self.n_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.up = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.down = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg["d_model"])
        self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, cos, sin, mask):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class Censer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "answer_id_tensor", torch.tensor(cfg["answer_ids"], dtype=torch.long), persistent=False
        )
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=cfg["pad_id"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.norm = RMSNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.score = nn.Linear(cfg["d_model"], len(cfg["answer_ids"]), bias=False)
        if cfg["tie_embeddings"]:
            self.lm_head.weight = self.embed.weight
        self._rope_cache = None

    def _rope(self, device, dtype):
        key = f"{device}|{dtype}"
        if self._rope_cache is None or self._rope_cache[2] != key:
            head_dim = self.cfg["d_model"] // self.cfg["n_heads"]
            cos, sin = build_rope_cache(self.cfg["max_seq_len"], head_dim, device, dtype, rope_scaling=self.cfg.get("rope_scaling"))
            self._rope_cache = (cos, sin, key)
        return self._rope_cache[0], self._rope_cache[1]

    def _attention_mask(self, input_ids):
        b, t = input_ids.shape
        pad = input_ids != self.cfg["pad_id"]
        causal = torch.ones(t, t, dtype=torch.bool, device=input_ids.device).tril()
        return pad[:, None, None, :] & causal[None, None, :, :]

    def _hidden(self, input_ids):
        cos, sin = self._rope(input_ids.device, self.embed.weight.dtype)
        mask = self._attention_mask(input_ids)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.norm(x)

    def forward(self, input_ids):
        return self.lm_head(self._hidden(input_ids))

    @torch.no_grad()
    def classify_probs(self, input_ids, temperature: float = 1.0):
        """Answer-head probabilities: softmax for binary heads, sigmoid for multi-label.
        `temperature` scales the logits (calibrated on val; stored in config.json)."""
        hidden = self._hidden(input_ids)
        non_pad = (input_ids != self.cfg["pad_id"]).sum(dim=1).clamp(min=1)
        idx = (non_pad - 1)[:, None, None].expand(-1, 1, hidden.shape[-1])
        logits = self.score(hidden.gather(1, idx)[:, 0, :].float()) / max(temperature, 1e-6)
        if logits.shape[-1] == 2:
            return logits.softmax(-1)
        return logits.sigmoid()

    @torch.no_grad()
    def classify(self, input_ids):
        return self.classify_probs(input_ids)[:, 1]



_STOPWORDS = {
    "en": {"the","and","you","is","to","a","in","that","i","of","for","it","my","me","do","what","how","are"},
    "cs": {"je","se","na","že","to","s","v","jsem","můj","moje","jak","co","nebo","pro","si","ty","ten"},
    "de": {"der","die","das","und","ist","ich","du","nicht","mit","ein","eine","auf","für","wie","was","mein","sich"},
    "es": {"el","la","de","que","y","en","un","una","es","por","con","para","mi","yo","como","qué","los","las"},
    "fr": {"le","la","de","et","les","des","en","un","une","est","pour","que","qui","je","tu","mon","ma","comment"},
    "it": {"il","lo","la","di","e","che","un","una","è","per","con","non","mi","io","come","sono","gli","le"},
    "pt": {"o","a","de","e","que","um","uma","em","para","com","não","eu","meu","minha","como","os","as"},
}

def detect_lang(text: str) -> str:
    """Heuristic language guess (scripts first, then Latin stopwords)."""
    import unicodedata, re as _re
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "CYRILLIC" in name:
                return "uk" if any(c in text for c in "їєії") else "ru"
            if "HAN" in name:
                return "zh"
            if "HIRAGANA" in name or "KATAKANA" in name:
                return "ja"
            if "HANGUL" in name:
                return "ko"
            if "ARABIC" in name:
                return "ar"
            if "DEVANAGARI" in name:
                return "hi"
            if "HEBREW" in name:
                return "he"
            break
    words = _re.findall(r"[a-zà-ÿ']+", text.lower())
    best, bn = "en", 0
    for lang, stops in _STOPWORDS.items():
        n = sum(1 for w in words if w in stops)
        if n > bn:
            best, bn = lang, n
    return best if bn > 0 else "en"



def load_state_dict_any(directory) -> dict:
    """Load model weights from model_int8.safetensors (dequantized) if present,
    else model_fp16.safetensors, else model.safetensors."""
    from pathlib import Path as _P
    from safetensors.torch import load_file

    d = _P(directory)
    if (d / "model_int8.safetensors").exists():
        st = load_file(str(d / "model_int8.safetensors"))
        out = {}
        for k, v in st.items():
            if k.endswith(".q"):
                base = k[:-2]
                out[base] = (v.float() * st[base + ".scale"]).to(torch.float32)
            elif not k.endswith(".scale"):
                out[k] = v
        return out
    if (d / "model_fp16.safetensors").exists():
        return {k: v.float() for k, v in load_file(str(d / "model_fp16.safetensors")).items()}
    return load_file(str(d / "model.safetensors"))


def remap_hf_state(state: dict) -> dict:
    """Standard Llama-layout safetensors -> training-layout state dict."""
    import re

    out = {"embed.weight": state["model.embed_tokens.weight"]}
    layer_ids = sorted({int(m.group(1)) for k in state if (m := re.match(r"model\.layers\.(\d+)\.", k))})
    for i in layer_ids:
        p, b = f"model.layers.{i}.", f"blocks.{i}."
        qkv = torch.cat(
            [state[p + "self_attn.q_proj.weight"], state[p + "self_attn.k_proj.weight"], state[p + "self_attn.v_proj.weight"]],
            dim=0,
        )
        out[b + "attn.qkv.weight"] = qkv.contiguous()
        out[b + "attn.proj.weight"] = state[p + "self_attn.o_proj.weight"]
        out[b + "mlp.gate.weight"] = state[p + "mlp.gate_proj.weight"]
        out[b + "mlp.up.weight"] = state[p + "mlp.up_proj.weight"]
        out[b + "mlp.down.weight"] = state[p + "mlp.down_proj.weight"]
        out[b + "norm1.weight"] = state[p + "input_layernorm.weight"]
        out[b + "norm2.weight"] = state[p + "post_attention_layernorm.weight"]
    out["norm.weight"] = state["model.norm.weight"]
    if "score.weight" in state:
        out["score.weight"] = state["score.weight"]
    return out


def internal_cfg(cfg: dict) -> dict:
    """config.json (Llama layout) -> the dict the Censer module expects."""
    return {
        "vocab_size": cfg["vocab_size"],
        "n_layers": cfg["num_hidden_layers"],
        "n_heads": cfg["num_attention_heads"],
        "d_model": cfg["hidden_size"],
        "d_ff": cfg["intermediate_size"],
        "max_seq_len": cfg["max_position_embeddings"],
        "dropout": cfg.get("dropout", 0.0),
        "tie_embeddings": True,
        "pad_id": cfg.get("pad_token_id", 0),
        "answer_ids": cfg["answer_ids"],
        "max_chars": cfg.get("max_chars", 4000),
        "rope_scaling": cfg.get("rope_scaling") or (cfg.get("rope_parameters") if cfg.get("rope_parameters", {}).get("rope_type") == "linear" else None),
        "categories": [cfg["id2label"][str(i)] for i in range(2, cfg.get("num_labels", 2))] if cfg.get("num_labels", 2) > 2 else [],
        "cat_thresholds": cfg.get("cat_thresholds") or {},
        "temperature": float(cfg.get("temperature", 1.0) or 1.0),
        "lang_thresholds": cfg.get("lang_thresholds") or {},
    }


class CenserPipeline:
    """Text in, "1" (unsafe) or "0" (safe) out."""

    def __init__(self, directory: str | Path = ".", threshold: float = 0.5, device: str | None = None, prefer_quant: bool = False) -> None:
        directory = Path(directory)
        self.cfg = internal_cfg(json.loads((directory / "config.json").read_text(encoding="utf-8")))
        self.tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.pad_id = self.tok.token_to_id(PAD)
        self.bos_id = self.tok.token_to_id(BOS)
        self.user_id = self.tok.token_to_id(USER)
        self.assistant_id = self.tok.token_to_id(ASSISTANT)
        self.threshold = threshold
        self.cat_thresholds = dict(self.cfg.get("cat_thresholds") or {})
        self.temperature = float(self.cfg.get("temperature", 1.0) or 1.0)
        self.lang_thresholds = dict(self.cfg.get("lang_thresholds") or {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Censer(self.cfg)
        if prefer_quant and (directory / "model_int8.safetensors").exists():
            state = remap_hf_state(load_state_dict_any(directory))
        else:
            from safetensors.torch import load_file
            state = remap_hf_state(load_file(directory / "model.safetensors"))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [k for k in missing if k != "lm_head.weight"]  # tied to embed.weight
        if missing or unexpected:
            raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
        self.model.to(self.device).eval()
        self.max_chars = self.cfg["max_chars"]

    def _encode(self, text: str) -> list[int]:
        body = self.tok.encode(text[: self.max_chars], add_special_tokens=False).ids
        ids = [self.bos_id, self.user_id, *body, self.assistant_id]
        return ids[: self.cfg["max_seq_len"]]

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def _encode_full(self, text: str) -> list[int]:
        body = self.tok.encode(text[: self.max_chars * 8], add_special_tokens=False).ids
        return [self.bos_id, self.user_id, *body, self.assistant_id]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        # long inputs: split into overlapping token chunks, aggregate by max
        out: list[dict] = []
        chunk_tokens, stride = 256, 192
        for text in texts:
            thr = self.threshold_for(text)
            ids = self._encode_full(text)
            body = ids[2:-1]
            if len(body) <= chunk_tokens:
                out.append(self._classify_ids([ids[: self.cfg["max_seq_len"]]], [thr])[0])
                continue
            chunks = []
            start = 0
            while start < len(body):
                chunks.append(body[start:start + chunk_tokens])
                if start + chunk_tokens >= len(body):
                    break
                start += stride
            rows = []
            for piece in chunks:
                cids = [self.bos_id, self.user_id, *piece, self.assistant_id]
                rows.append(self._classify_ids([cids[: self.cfg["max_seq_len"]]], [thr])[0])
            best = max(rows, key=lambda r: r["score"])
            # aggregate categories by max prob
            n_cat = len(rows[0].get("category_scores", []))
            cat_scores = [max(r["category_scores"][i] for r in rows) for i in range(n_cat)] if n_cat else []
            best["category_scores"] = [round(c, 4) for c in cat_scores]
            cats = self.categories()
            best["categories"] = ([c for c, p in zip(cats, cat_scores) if p >= float(self.cat_thresholds.get(c, self.threshold))]
                                  if best["unsafe"] else [])
            best["chunks"] = len(chunks)
            out.append(best)
        return out

    def threshold_for(self, text: str) -> float:
        lang = detect_lang(text)
        return float(self.lang_thresholds.get(lang, self.threshold))

    def categories(self) -> list[str]:
        return self.cfg.get("categories") or []

    def _classify_ids(self, seqs: list[list[int]], thresholds: list[float] | None = None) -> list[dict]:
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq)
        probs = self.model.classify_probs(ids.to(self.device), temperature=self.temperature).float().cpu().tolist()
        if thresholds is None:
            thresholds = [self.threshold] * len(seqs)
        res = []
        for i, row in enumerate(probs):
            p = row[1]
            cats = self.categories()
            cat_scores = row[2:] if cats else []
            is_unsafe = p >= thresholds[i]
            res.append({
                "label": "1" if is_unsafe else "0",
                "unsafe": is_unsafe,
                "score": round(p, 4),
                "category_scores": [round(c, 4) for c in cat_scores],
                # categories are only reported when the binary verdict is unsafe
                "categories": ([c for c, sc in zip(cats, cat_scores)
                                if sc >= float(self.cat_thresholds.get(c, self.threshold))]
                               if is_unsafe else []),
            })
        return res


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def repl(pipe: "CenserPipeline") -> int:
    """Small interactive terminal UI: phrase in, safe/unsafe + score out."""
    print("Censer.9 - interactive terminal UI")
    print("Type a phrase, Enter to classify. 'exit' to quit.\n")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        text = line.strip()
        if text.lower() in {"exit", "quit", ":q"}:
            return 0
        if not text:
            continue
        v = pipe.predict(text)
        label = "unsafe" if v["unsafe"] else "safe"
        colored = _color(f"{label:6s}", "31" if v["unsafe"] else "32")
        print(f"{colored} {v['score']:.4f} | {text}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="censer.9",
        description="Censer.9 safety classifier (standalone). Run without text to open the interactive UI.",
    )
    parser.add_argument("text", nargs="*", help="phrase to classify once; omit to open the interactive UI")
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--prefer-quant", action="store_true",
                        help="load model_int8.safetensors (0.9 MB) instead of fp32 if present")
    parser.add_argument("--repl", action="store_true", help="interactive UI (default when no text is given)")
    args = parser.parse_args(argv)

    pipe = CenserPipeline(args.model_dir, args.threshold, prefer_quant=args.prefer_quant)
    text = " ".join(args.text).strip()
    if args.repl or not text:
        raise SystemExit(repl(pipe))
    verdict = pipe.predict(text)
    print(json.dumps(verdict) if args.json else verdict["label"])


if __name__ == "__main__":
    main()
