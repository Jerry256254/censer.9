from __future__ import annotations

import json
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.processors import TemplateProcessing

PAD, BOS, USER, ASSISTANT = "[PAD]", "[BOS]", "[USER]", "[ASSISTANT]"
SPECIAL_TOKENS = [PAD, BOS, USER, ASSISTANT]
PROMPT_TEMPLATE = "{user} {assistant}"


class KucTokenizer:
    def __init__(self, tok: Tokenizer) -> None:
        from .categories import ANSWER_CHARS

        self._tok = tok
        self.answer_ids = [tok.token_to_id(c) for c in ANSWER_CHARS]
        self.pad_id = tok.token_to_id(PAD)
        self.bos_id = tok.token_to_id(BOS)
        self.user_id = tok.token_to_id(USER)
        self.assistant_id = tok.token_to_id(ASSISTANT)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode_prompt(self, text: str, max_len: int) -> list[int]:
        body = self._tok.encode(text, add_special_tokens=False).ids
        ids = [self.bos_id, self.user_id, *body, self.assistant_id]
        return ids[:max_len]

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path / "tokenizer.json"))
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "special_tokens": SPECIAL_TOKENS,
                    "answer_ids": self.answer_ids,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, directory: str | Path) -> "KucTokenizer":
        path = Path(directory)
        tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        _install_post_processor(tok)
        return cls(tok)


def _install_post_processor(tok: Tokenizer) -> None:
    tok.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[(BOS, tok.token_to_id(BOS)), (PAD, tok.token_to_id(PAD))],
    )


def train_tokenizer(
    texts: list[str],
    vocab_size: int,
    min_frequency: int,
    out_dir: str | Path,
) -> KucTokenizer:
    corpus_path = Path(out_dir) / "corpus.txt"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8") as handle:
        for text in texts:
            clean = text.replace("\n", " ").strip()
            if clean:
                handle.write(clean + "\n")

    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )

    tok = bpe._tokenizer
    _install_post_processor(tok)
    corpus_path.unlink(missing_ok=True)

    wrapper = KucTokenizer(tok)
    wrapper.save(out_dir)
    return wrapper
