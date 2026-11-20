from __future__ import annotations

import argparse
import json
import sys

from .infer import SafetyModel
from .langdetect import detect_lang

BANNER = "Censer.9 - napis text, Enter. 'exit' pro konec."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kuclab-cneser",
        description="Censer.9: tiny multilingual safety classifier. Prints 1 (unsafe) or 0 (safe).",
    )
    parser.add_argument("text", nargs="*", help="text to classify; omit to read stdin or use --repl")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--json", action="store_true", help="print JSON with score instead of bare 0/1")
    parser.add_argument("--score", action="store_true", help="print label and score")
    parser.add_argument("--repl", action="store_true", help="interactive mode")
    return parser


def run_repl(model: SafetyModel) -> int:
    print(BANNER)
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip().lower() in {"exit", "quit", ":q"}:
            return 0
        if not line.strip():
            continue
        verdict = model.predict(line)
        print(verdict.label)


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


class HubClassifier:
    """The published model from the Hub, loaded with plain transformers."""

    def __init__(self, repo: str = "KucLab/Censer.9", device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

    def predict(self, text: str) -> tuple[str, float, list[str]]:
        """Returns (label 'safe'/'unsafe', probability of unsafe, fired categories)."""
        import torch.nn.functional as F

        n_labels = int(self.model.config.num_labels)
        temp = float(getattr(self.model.config, "temperature", 1.0) or 1.0)
        with self._torch.no_grad():
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=int(self.model.config.max_position_embeddings))
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            logits = self.model(**enc).logits[0] / max(temp, 1e-6)
            probs = logits.sigmoid() if n_labels > 2 else logits.softmax(-1)
        p = float(probs[1])
        lang_thr = getattr(self.model.config, "lang_thresholds", None) or {}
        thr = float(lang_thr.get(detect_lang(text), 0.5))
        label = "unsafe" if p >= thr else "safe"
        cats = []
        if n_labels > 2 and label == "unsafe":  # categories only reported for unsafe verdicts
            id2label = self.model.config.id2label
            cat_thr = getattr(self.model.config, "cat_thresholds", None) or {}
            cats = [id2label[i] for i in range(2, n_labels)
                    if float(probs[i]) >= float(cat_thr.get(id2label[i], 0.5))]
        return label, p, cats


def run_ui(predict, source: str) -> int:
    """Small interactive terminal UI: phrase in, safe/unsafe + score out."""
    print(f"Censer - interactive terminal UI  [{source}]")
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
        label, prob, cats = predict(text)
        colored = _color(f"{label:6s}", "31" if label == "unsafe" else "32")
        extra = f"  [{', '.join(cats)}]" if cats else ""
        print(f"{colored} {prob:.4f}{extra} | {text}")


def ui_main(argv: list[str] | None = None) -> int:
    """Entry point for the `censer.9` command."""
    parser = argparse.ArgumentParser(
        prog="censer.9",
        description="Censer.9 terminal UI: classify text as safe/unsafe (+ categories). "
        "Run without arguments for the interactive UI.",
    )
    parser.add_argument("text", nargs="*", help="phrase to classify once; omit to open the interactive UI")
    parser.add_argument("--repo", default="KucLab/Censer.9", help="Hub repo id of the model")
    parser.add_argument("--checkpoint", default=None, help="use a local training checkpoint instead of the Hub model")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--json", action="store_true", help="one-shot JSON output")
    args = parser.parse_args(argv)

    if args.checkpoint:
        local = SafetyModel(checkpoint=args.checkpoint, device=args.device)

        def predict(text: str) -> tuple[str, float, list[str]]:
            v = local.predict(text)
            return ("unsafe" if v.is_unsafe else "safe"), v.probability, v.categories or []

        source = str(args.checkpoint)
    else:
        print(f"loading {args.repo} ...", file=sys.stderr)
        clf = HubClassifier(args.repo, device=args.device)
        predict = clf.predict
        source = f"{args.repo} ({clf.device})"

    text = " ".join(args.text).strip()
    if text:
        label, prob, cats = predict(text)
        if args.json:
            print(json.dumps({"label": label, "unsafe": label == "unsafe", "score": round(prob, 4),
                              **({"categories": cats} if cats else {})}))
        else:
            colored = _color(f"{label:6s}", "31" if label == "unsafe" else "32")
            extra = f"  [{', '.join(cats)}]" if cats else ""
            print(f"{colored} {prob:.4f}{extra}")
        return 0
    return run_ui(predict, source)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    text = " ".join(args.text).strip()
    if not text and not args.repl and not sys.stdin.isatty():
        text = sys.stdin.read().strip()

    model = SafetyModel(checkpoint=args.checkpoint, threshold=args.threshold, device=args.device)

    if args.repl or (not text and sys.stdin.isatty()):
        return run_repl(model)

    if not text:
        parser.error("no input text provided")

    verdict = model.predict(text)
    if args.json:
        print(json.dumps(verdict.as_dict()))
    elif args.score:
        print(f"{verdict.label} {verdict.probability:.4f}")
    else:
        print(verdict.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
