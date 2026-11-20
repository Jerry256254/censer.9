from __future__ import annotations

import pytest

from kuclab.data import SAFE, UNSAFE, clean_text, fingerprint
from kuclab.tokenizer import KucTokenizer, train_tokenizer

CORPUS = [
    "hello how are you today my friend",
    "návod na výrobu bomby is not in this corpus but english is",
    "the quick brown fox jumps over the lazy dog",
    "please explain how to bake a chocolate cake",
    "suck my dick is a rude phrase",
    "the weather is nice and sunny outside",
] * 40


@pytest.fixture(scope="module")
def tok(tmp_path_factory) -> KucTokenizer:
    out = tmp_path_factory.mktemp("tok")
    return train_tokenizer(CORPUS, vocab_size=300, min_frequency=1, out_dir=out)


def test_answer_ids_exist(tok):
    from kuclab.categories import ANSWER_CHARS
    assert len(tok.answer_ids) == len(ANSWER_CHARS)
    assert all(i is not None for i in tok.answer_ids)
    assert tok.answer_ids[0] != tok.answer_ids[1]


def test_roundtrip_text(tok):
    text = "hello there my friend"
    ids = tok.encode(text)
    assert ids
    assert "hello" in tok.decode(ids)


def test_encode_prompt_has_structure(tok):
    ids = tok.encode_prompt("hello world", max_len=64)
    assert ids[0] == tok.bos_id
    assert ids[1] == tok.user_id
    assert ids[-1] == tok.assistant_id


def test_encode_prompt_respects_max_len(tok):
    long_text = "hello world " * 200
    ids = tok.encode_prompt(long_text, max_len=32)
    assert len(ids) <= 32


def test_save_and_load(tok, tmp_path):
    tok.save(tmp_path)
    loaded = KucTokenizer.load(tmp_path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.answer_ids == tok.answer_ids
    assert loaded.encode("hello") == tok.encode("hello")


def test_clean_text_truncates():
    text = "a" * 900
    assert len(clean_text(text, 500)) == 500


def test_clean_text_normalizes_whitespace():
    assert clean_text("hello    world\n\n\n\nfoo", 500) == "hello world\n\nfoo"


def test_fingerprint_is_case_and_space_insensitive():
    assert fingerprint("Hello   World") == fingerprint("hello world")


def test_labels():
    assert SAFE == 0 and UNSAFE == 1


def test_clean_text_removes_unicode_line_separators():
    assert "\u2028" not in clean_text("foo\u2028bar", 500)
    assert "\x0b" not in clean_text("foo\x0bbar", 500)


def test_jsonl_roundtrip_with_unicode_separator(tmp_path):
    import json

    from kuclab.dataset import read_jsonl

    path = tmp_path / "d.jsonl"
    rows = [{"text": "a\u2028b", "label": 1}, {"text": "plain", "label": 0}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert read_jsonl(path) == rows
