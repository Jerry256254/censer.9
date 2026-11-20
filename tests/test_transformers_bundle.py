"""The Hub bundle must be a standard transformers model — no custom code needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kuclab.checkpoint_utils import make_test_checkpoint  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

PROBES = [
    "hello there friend",
    "please bake a chocolate cake",
    "how to make a bomb at home",
    "suck my dick",
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from upload_hf import build_bundle

    tmp = tmp_path_factory.mktemp("hub")
    ckpt = make_test_checkpoint(tmp / "ckpt")
    out = tmp / "bundle"
    build_bundle(ckpt, out)
    return out, ckpt


def test_config_is_standard_llama(bundle):
    out, _ = bundle
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "llama"
    assert cfg["architectures"] == ["LlamaForSequenceClassification"]
    assert "auto_map" not in cfg
    from kuclab.categories import ID2LABEL
    assert cfg["num_labels"] == len(cfg["answer_ids"]) == 12
    assert cfg["id2label"] == {str(k): v for k, v in ID2LABEL.items()}
    assert cfg["problem_type"] == "multi_label_classification"
    assert cfg["rms_norm_eps"] == 1e-5
    assert cfg["rope_parameters"] == {"rope_type": "default", "rope_theta": 10000.0}


def test_weights_have_llama_layout(bundle):
    from safetensors.torch import load_file

    out, _ = bundle
    keys = set(load_file(out / "model.safetensors"))
    assert "model.embed_tokens.weight" in keys
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert "model.layers.0.self_attn.k_proj.weight" in keys
    assert "model.layers.0.self_attn.v_proj.weight" in keys
    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.norm.weight" in keys
    assert "score.weight" in keys
    assert not any(k.startswith(("embed.", "blocks.", "lm_head.")) for k in keys)


def test_score_head_is_untied_and_remaps_legacy(bundle):
    """Censer.9 uses a separate linear score head (not tied to embeddings);
    legacy tied checkpoints must load through remap_legacy_state."""
    from safetensors.torch import load_file

    from kuclab.model import remap_legacy_state

    out, ckpt = bundle
    state = load_file(out / "model.safetensors")
    cfg = json.loads((out / "config.json").read_text())
    ids = cfg["answer_ids"]
    assert state["score.weight"].shape == (len(ids), cfg["hidden_size"])
    # legacy checkpoint (tied head) remaps to an explicit score.weight built from embeddings
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    legacy = {"model": {k: v for k, v in raw["model"].items() if not k.startswith("score.")},
              "model_config": raw["model_config"]}
    remapped = remap_legacy_state(legacy)["model"]
    assert "score.weight" in remapped
    assert torch.equal(remapped["score.weight"], raw["model"]["embed.weight"][list(raw["model_config"]["answer_ids"])])


def test_qkv_split_matches_fused_qkv(bundle):
    """The first/second/third slices of the fused qkv must land in q/k/v."""
    import torch as t

    out, ckpt = bundle
    from safetensors.torch import load_file

    state = load_file(out / "model.safetensors")
    raw = t.load(ckpt, map_location="cpu", weights_only=False)["model"]
    d = raw["blocks.0.attn.qkv.weight"].shape[1]
    qkv = raw["blocks.0.attn.qkv.weight"]
    assert t.equal(state["model.layers.0.self_attn.q_proj.weight"], qkv[:d])
    assert t.equal(state["model.layers.0.self_attn.k_proj.weight"], qkv[d : 2 * d])
    assert t.equal(state["model.layers.0.self_attn.v_proj.weight"], qkv[2 * d : 3 * d])


def test_tokenizer_adds_prompt_template(bundle):
    from tokenizers import Tokenizer

    out, _ = bundle
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    ids = tok.encode("hello there").ids
    assert ids[0] == tok.token_to_id("[BOS]")
    assert ids[1] == tok.token_to_id("[USER]")
    assert ids[-1] == tok.token_to_id("[ASSISTANT]")


def test_loads_without_trust_remote_code(bundle):
    out, _ = bundle
    model = transformers.AutoModelForSequenceClassification.from_pretrained(str(out))
    assert type(model.config).__name__ == "LlamaConfig"
    assert isinstance(model, transformers.PreTrainedModel)


def test_transformers_matches_repo_predictions(bundle):
    out, ckpt = bundle
    from kuclab.infer import SafetyModel

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo_model = SafetyModel(ckpt)
    tok = AutoTokenizer.from_pretrained(str(out))
    hub_model = AutoModelForSequenceClassification.from_pretrained(str(out)).eval()
    temp = max(float(repo_model.temperature), 1e-6)

    for text in PROBES:
        with torch.no_grad():
            enc = tok(text, return_tensors="pt")
            logits = hub_model(**enc).logits[0]
            prob = (logits / temp).sigmoid()[1].item() if logits.shape[0] > 2 else (logits / temp).softmax(-1)[1].item()
        assert abs(prob - repo_model.predict(text).probability) < 1e-3


def test_pipeline_without_trust_remote_code(bundle):
    from transformers import pipeline

    from kuclab.infer import SafetyModel

    out, ckpt = bundle
    repo_model = SafetyModel(ckpt)
    clf = pipeline("text-classification", model=str(out))
    temp = max(float(repo_model.temperature), 1e-6)
    for text in PROBES:
        res = clf(text, top_k=-1)  # multi-label head: all 12 scores
        if isinstance(res, dict):
            res = [res]
        scores = {r["label"]: r["score"] for r in res}
        assert "unsafe" in scores
        # hub pipeline returns raw sigmoid; repo applies calibrated temperature
        import math
        expected = repo_model.predict(text).probability
        raw = math.log(expected / (1 - expected)) * temp  # undo temp -> raw logit
        expected_raw = 1 / (1 + math.exp(-raw))
        assert abs(scores["unsafe"] - expected_raw) < 1e-3


def test_batched_pipeline_matches_single(bundle):
    from transformers import pipeline

    out, _ = bundle
    clf = pipeline("text-classification", model=str(out))
    batch = clf(PROBES, top_k=-1)
    singles = [clf(t, top_k=-1) for t in PROBES]
    if isinstance(batch[0], dict):
        batch = [[b] for b in batch]
        singles = [[s] for s in singles]
    for b, s in zip(batch, singles):
        bs = {r["label"]: r["score"] for r in b}
        ss = {r["label"]: r["score"] for r in s}
        assert bs.keys() == ss.keys()
        for k in bs:
            assert abs(bs[k] - ss[k]) < 1e-4
