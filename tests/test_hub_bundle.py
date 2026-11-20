"""The standalone censer.py shipped to the Hub must match the training code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kuclab.checkpoint_utils import make_test_checkpoint  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_standalone():
    spec = importlib.util.spec_from_file_location("censer_standalone", SCRIPTS / "censer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["censer_standalone"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    ckpt = make_test_checkpoint(tmp / "ckpt")
    sys.path.insert(0, str(SCRIPTS))
    from upload_hf import build_bundle

    out = tmp / "bundle"
    build_bundle(ckpt, out)
    return out, ckpt


def test_bundle_has_all_required_files(bundle):
    out, _ = bundle
    assert {"model.safetensors", "config.json", "tokenizer.json", "censer.py", "censer.9"} <= set(
        p.name for p in out.iterdir()
    )


def test_censer9_launcher_is_executable_shebang_script(bundle):
    out, _ = bundle
    launcher = out / "censer.9"
    content = launcher.read_text()
    assert content.startswith("#!")
    assert "from censer import main" in content
    assert launcher.stat().st_mode & 0o111, "censer.5 must be executable in the bundle"


def test_standalone_repl_classifies_until_exit(bundle, monkeypatch, capsys):
    out, _ = bundle
    standalone = load_standalone()
    answers = iter(["hello there friend", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    with pytest.raises(SystemExit) as exc:
        standalone.main(["--model-dir", str(out)])
    captured = capsys.readouterr().out
    assert exc.value.code == 0
    assert "interactive terminal UI" in captured
    assert "safe" in captured and "hello there friend" in captured


def test_config_matches_model(bundle):
    import json

    out, _ = bundle
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "llama"
    assert cfg["answer_ids"] and len(cfg["answer_ids"]) == 12  # 0/1 + 10 categories
    assert cfg["vocab_size"] > cfg["answer_ids"][1]
    assert cfg["max_chars"] > 0
    assert cfg.get("pad_token_id", cfg.get("pad_id")) >= 0


def test_standalone_matches_repo_predictions(bundle):
    out, ckpt = bundle
    standalone = load_standalone()
    from kuclab.infer import SafetyModel

    texts = ["hello there friend", "please bake a chocolate cake", "thanks a lot"]
    a = standalone.CenserPipeline(out)
    b = SafetyModel(ckpt)
    for text in texts:
        assert abs(a.predict(text)["score"] - b.predict(text).probability) < 1e-4


def test_standalone_labels_are_bare(bundle):
    out, _ = bundle
    standalone = load_standalone()
    pipe = standalone.CenserPipeline(out)
    assert pipe.predict("hello")["label"] in {"0", "1"}


def test_standalone_batch_shape(bundle):
    out, _ = bundle
    standalone = load_standalone()
    pipe = standalone.CenserPipeline(out)
    verdicts = pipe.predict_batch(["hello", "how are you", "thanks"])
    assert len(verdicts) == 3
    assert all("score" in v and "label" in v for v in verdicts)
