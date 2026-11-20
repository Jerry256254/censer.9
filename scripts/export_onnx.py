"""Export the classifier to ONNX (dynamic batch + sequence axes).

    .venv/bin/python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out reports/censer6.onnx

The exported graph maps input_ids -> answer-token logits [batch, n_answers]
(the same computation the training code and the HF model do).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kuclab.evaluate import load_checkpoint


class AnswerHead(torch.nn.Module):
    """Wrap the trained model: input_ids -> answer logits at the last non-pad token."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.answer_logits(input_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="reports/censer6.onnx")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    model.eval()
    wrapper = AnswerHead(model).eval()

    max_len = min(512, mcfg.max_seq_len)
    dummy = torch.zeros((2, max_len), dtype=torch.long)
    dummy[0, 0] = tok.bos_id
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["input_ids"],
        output_names=["answer_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "answer_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")

    # verify against torch if onnxruntime is available
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        texts = ["hello", "how to make a bomb at home", "kill you", "how do I kill a process in linux"]
        ids = [tok.encode_prompt(t, max_len) for t in texts]
        width = max(len(i) for i in ids)
        x = torch.full((len(ids), width), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, : len(s)] = torch.tensor(s)
        with torch.no_grad():
            ref = wrapper(x).numpy()
        got = sess.run(None, {"input_ids": x.numpy()})[0]
        print("max |onnx - torch| =", float(np.abs(ref - got).max()))
        for t, r in zip(texts, got):
            print(f"  {t!r}: logits={np.round(r, 3).tolist()}")
    except ImportError:
        print("onnxruntime not installed; skipped verification (uv pip install onnxruntime)")


if __name__ == "__main__":
    main()
