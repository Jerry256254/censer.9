"""Upload the trained model to the Hugging Face Hub.

The checkpoint is exported as a *standard* ``LlamaForSequenceClassification``
(the Censer architecture is exactly Llama: RoPE, RMSNorm, SwiGLU, tied
embeddings, fused QKV). The classification head ``score`` is initialised from
the embedding rows of the answer tokens "0"/"1", so predictions are identical
to the training code — and the model loads with plain ``transformers``, no
``trust_remote_code`` and no custom files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_CARD = (Path(__file__).resolve().parent / "MODEL_CARD.md").read_text(encoding="utf-8")


def cat_thresholds() -> dict:
    """Per-category calibrated thresholds from reports/calibration.json.

    Categories with no positive support in catval keep the default 0.5."""
    import json as _json

    p = Path(__file__).resolve().parent.parent / "reports" / "calibration.json"
    out = {}
    if p.exists():
        try:
            cal = _json.loads(p.read_text())
            for cat, v in cal.get("categories", {}).items():
                thr = float(v.get("best_threshold", 0.5))
                f1 = float(v.get("f1_at_best", 0.0))
                out[cat] = thr if f1 > 0 else 0.5
        except Exception:
            pass
    return out


def calibration_temperature() -> float:
    """Binary temperature from reports/calibration.json (val-scaled; default 1.0)."""
    import json as _json

    p = Path(__file__).resolve().parent.parent / "reports" / "calibration.json"
    if p.exists():
        try:
            return float(_json.loads(p.read_text()).get("temperature", 1.0))
        except Exception:
            pass
    return 1.0


def lang_thresholds() -> dict:
    """Per-language binary thresholds from reports/multilingual.json (textdetox best_thr)."""
    import json as _json

    p = Path(__file__).resolve().parent.parent / "reports" / "multilingual.json"
    out = {}
    if p.exists():
        try:
            res = _json.loads(p.read_text()).get("results", {})
            for k, v in res.items():
                # only trust a tuned threshold when the language actually performs
                # (best_f1 >= 0.5); otherwise keep the safe default 0.5 so that
                # unsupported languages are not flagged wholesale
                if (k.startswith("textdetox_") and v.get("best_thr") is not None
                        and v.get("n", 0) >= 400 and float(v.get("best_f1", 0)) >= 0.5):
                    out[k.split("_", 1)[1]] = round(float(v["best_thr"]), 3)
        except Exception:
            pass
    return out or {"en": 0.5}


def to_llama_state_dict(state: dict, answer_ids: list[int]) -> dict:
    """Map the training checkpoint onto standard LlamaForSequenceClassification keys."""
    model = state["model"]
    cfg = state["model_config"]
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]

    out = {"model.embed_tokens.weight": model["embed.weight"].contiguous()}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        b = f"blocks.{i}."
        qkv = model[f"{b}attn.qkv.weight"]
        out[f"{p}self_attn.q_proj.weight"] = qkv[:d].contiguous()
        out[f"{p}self_attn.k_proj.weight"] = qkv[d : 2 * d].contiguous()
        out[f"{p}self_attn.v_proj.weight"] = qkv[2 * d : 3 * d].contiguous()
        out[f"{p}self_attn.o_proj.weight"] = model[f"{b}attn.proj.weight"].contiguous()
        out[f"{p}mlp.gate_proj.weight"] = model[f"{b}mlp.gate.weight"].contiguous()
        out[f"{p}mlp.up_proj.weight"] = model[f"{b}mlp.up.weight"].contiguous()
        out[f"{p}mlp.down_proj.weight"] = model[f"{b}mlp.down.weight"].contiguous()
        out[f"{p}input_layernorm.weight"] = model[f"{b}norm1.weight"].contiguous()
        out[f"{p}post_attention_layernorm.weight"] = model[f"{b}norm2.weight"].contiguous()
    out["model.norm.weight"] = model["norm.weight"].contiguous()
    # The answer head is a separate 12x128 (or 2x128) matrix in .8+; older tied
    # checkpoints are remapped via remap_legacy_state before export.
    if "score.weight" in model:
        out["score.weight"] = model["score.weight"].clone().contiguous()
    else:
        out["score.weight"] = model["embed.weight"][list(answer_ids)].clone().contiguous()
    return out


def llama_config(state: dict, tok_meta: dict, tok_ids: dict) -> dict:
    mc = dict(state["model_config"])
    mc.pop("answer_ids", None)
    from kuclab.categories import ID2LABEL

    n_answers = len(tok_meta["answer_ids"])
    rope_scaling = mc.get("rope_scaling") or None
    rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
    if rope_scaling and float(rope_scaling.get("factor", 1.0)) > 1.0:
        rope_parameters = {
            "rope_type": "linear",
            "factor": float(rope_scaling["factor"]),
            "rope_theta": 10000.0,
        }
    id2label = {str(k): v for k, v in ID2LABEL.items()} if n_answers > 2 else {"0": "safe", "1": "unsafe"}
    return {
        "architectures": ["LlamaForSequenceClassification"],
        "model_type": "llama",
        "num_labels": n_answers,
        "id2label": id2label,
        "label2id": {v: int(k) for k, v in id2label.items()},
        **({"problem_type": "multi_label_classification"} if n_answers > 2 else {}),
        "vocab_size": mc["vocab_size"],
        "hidden_size": mc["d_model"],
        "intermediate_size": mc["d_ff"],
        "num_hidden_layers": mc["n_layers"],
        "num_attention_heads": mc["n_heads"],
        "num_key_value_heads": mc["n_heads"],
        "head_dim": mc["d_model"] // mc["n_heads"],
        "max_position_embeddings": mc["max_seq_len"],
        "rms_norm_eps": 1e-5,
        "rope_parameters": rope_parameters,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "pad_token_id": mc["pad_id"],
        "bos_token_id": tok_ids["[BOS]"],
        "eos_token_id": tok_ids["[ASSISTANT]"],
        # Censer-specific metadata, kept for the standalone censer.py:
        "name": state.get("name", "Censer.9"),
        "max_chars": int(state.get("max_chars", 4000)),
        "answer_ids": list(tok_meta["answer_ids"]),
        "special_tokens": list(tok_meta["special_tokens"]),
        "threshold": 0.5,
        # calibrated per-category cutoffs (tuned on held-out catval; falls back to 0.5)
        **({"cat_thresholds": cat_thresholds()} if n_answers > 2 else {}),
        "temperature": calibration_temperature(),
        "lang_thresholds": lang_thresholds(),
    }


def write_tokenizer(src: Path, dst: Path) -> dict:
    """Ship the tokenizer with the prompt template baked in, so AutoTokenizer
    produces exactly [BOS][USER] text [ASSISTANT]. Returns the special-token ids."""
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing

    tok = Tokenizer.from_file(str(src))
    ids = {name: tok.token_to_id(name) for name in ("[PAD]", "[BOS]", "[USER]", "[ASSISTANT]")}
    tok.post_processor = TemplateProcessing(
        single="[BOS] [USER] $A [ASSISTANT]",
        special_tokens=[("[BOS]", ids["[BOS]"]), ("[USER]", ids["[USER]"]), ("[ASSISTANT]", ids["[ASSISTANT]"])],
    )
    tok.save(str(dst))
    return ids


def write_tokenizer_config(dst: Path, ids: dict) -> None:
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[ASSISTANT]",
        "model_max_length": 4096,
    }
    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def build_bundle(ckpt: Path, out: Path) -> None:
    """Write a self-contained, standard-transformers model directory."""
    import torch
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)

    tok_dir = ckpt.parent / "tokenizer"
    tok_meta = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))
    answer_ids = list(tok_meta["answer_ids"])

    tensors = to_llama_state_dict(state, answer_ids)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    ids = write_tokenizer(tok_dir / "tokenizer.json", out / "tokenizer.json")
    write_tokenizer_config(out / "tokenizer_config.json", ids)

    config = llama_config(state, tok_meta, ids)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__).parent / "censer.py", out / "censer.py")
    launcher = Path(__file__).parent / "censer.9"
    shutil.copy2(launcher, out / "censer.9")
    os.chmod(out / "censer.9", 0o755)
    (out / "README.md").write_text(MODEL_CARD, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Censer.9 to the Hugging Face Hub.")
    parser.add_argument("--repo", default="KucLab/Censer.9", help="e.g. KucLab/Censer.9")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        build_bundle(ckpt, bundle)
        print(f"bundle: {[p.name for p in sorted(bundle.iterdir())]}")

        if args.dry_run:
            print("dry run, not uploading")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        who = api.whoami()
        print(f"authenticated as: {who.get('name')}")

        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(bundle),
            repo_id=args.repo,
            repo_type="model",
            commit_message="Standard LlamaForSequenceClassification layout: loads with plain transformers, no trust_remote_code",
            delete_patterns=["modeling_censer.py"],
        )

        figs = Path(__file__).resolve().parent.parent / "reports" / "figs"
        if figs.exists():
            api.upload_folder(
                folder_path=str(figs),
                repo_id=args.repo,
                repo_type="model",
                path_in_repo="figs",
                commit_message="Add comparison charts vs other safety classifiers",
            )
        reports = Path(__file__).resolve().parent.parent / "reports"
        if reports.exists():
            jsons = sorted(p for p in reports.iterdir() if p.suffix in {".json", ".md"})
            for p in jsons:
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=f"reports/{p.name}",
                    repo_id=args.repo,
                    repo_type="model",
                    commit_message=f"Add report {p.name}",
                )
        print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
