from __future__ import annotations

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from kuclab.checkpoint_utils import make_test_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    out = tmp_path_factory.mktemp("ckpt")
    path = make_test_checkpoint(out)
    return path


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kuclab.cli", *args],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_prints_bare_label(checkpoint):
    result = run_cli(["hello how are you", "--checkpoint", str(checkpoint), "--device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {"0", "1"}
    assert len(result.stdout.strip()) == 1


def test_cli_json_mode(checkpoint):
    result = run_cli(["hello", "--checkpoint", str(checkpoint), "--device", "cpu", "--json"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["label"] in {"0", "1"}
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["unsafe"] == (payload["label"] == "1")


def test_cli_score_mode(checkpoint):
    result = run_cli(["hello", "--checkpoint", str(checkpoint), "--device", "cpu", "--score"])
    assert result.returncode == 0, result.stderr
    label, score = result.stdout.split()
    assert label in {"0", "1"}
    assert 0.0 <= float(score) <= 1.0


def test_cli_long_input_is_truncated(checkpoint):
    result = run_cli(["word " * 400, "--checkpoint", str(checkpoint), "--device", "cpu"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {"0", "1"}


def test_threshold_is_respected(checkpoint):
    hot = run_cli(["hello", "--checkpoint", str(checkpoint), "--device", "cpu", "--threshold", "0.0"])
    cold = run_cli(["hello", "--checkpoint", str(checkpoint), "--device", "cpu", "--threshold", "1.01"])
    assert hot.stdout.strip() == "1"
    assert cold.stdout.strip() == "0"


def run_ui(args: list[str], stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "import sys; from kuclab.cli import ui_main; sys.exit(ui_main(sys.argv[1:]))", *args],
        capture_output=True,
        text=True,
        input=stdin,
        timeout=180,
    )


def test_censer9_script_is_registered():
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["censer.9"] == "kuclab.cli:ui_main"


def test_ui_one_shot_with_checkpoint(checkpoint):
    result = run_ui(["hello how are you", "--checkpoint", str(checkpoint), "--device", "cpu", "--json"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["label"] in {"safe", "unsafe"}
    assert payload["unsafe"] == (payload["label"] == "unsafe")
    assert 0.0 <= payload["score"] <= 1.0


def test_ui_repl_reads_phrases_until_exit(checkpoint):
    result = run_ui(
        ["--checkpoint", str(checkpoint), "--device", "cpu"],
        stdin="hello there\nhow to make a bomb\nexit\n",
    )
    assert result.returncode == 0, result.stderr
    assert "interactive terminal UI" in result.stdout
    lines = [l for l in result.stdout.splitlines() if " | " in l]
    assert len(lines) == 2
    assert all(any(tok in {"safe", "unsafe"} for tok in l.split()) for l in lines)
    assert "hello there" in lines[0] and "how to make a bomb" in lines[1]
