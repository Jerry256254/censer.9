"""Export quantized variants of the HF bundle: fp16 safetensors + int8 (dynamic, CPU).

    .venv/bin/python scripts/export_quant.py --bundle /tmp/bundle

Writes model_fp16.safetensors and model_int8.pt (torch dynamic-quantized Linear)
into the bundle dir, and prints sizes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    bundle = Path(args.bundle)

    from safetensors.torch import load_file, save_file

    state = load_file(bundle / "model.safetensors")
    fp16 = {k: v.to(torch.float16) for k, v in state.items()}
    save_file(fp16, str(bundle / "model_fp16.safetensors"), metadata={"format": "pt"})
    mb16 = (bundle / "model_fp16.safetensors").stat().st_size / 1e6

    # int8 dynamic quantization on CPU: quantize every Linear weight
    import torch.nn as nn

    class Quantizable(nn.Module):
        pass

    # quantize raw weight tensors via dynamic quantization of matmul is not
    # possible on raw tensors; instead build a tiny module map and use
    # torch.ao.quantization.quantize_dynamic on a wrapper model is overkill.
    # Practical approach: per-tensor int8 symmetric quantization of 2D weights,
    # dequantize at load is done by the consumer; here we store int8 weights +
    # scales in a safetensors-compatible sidecar and verify roundtrip error.
    import json

    int8_state = {}
    scales = {}
    for k, v in state.items():
        if v.dim() == 2 and v.dtype == torch.float32:
            scale = v.abs().max() / 127.0
            if float(scale) == 0:
                scale = torch.tensor(1.0)
            q = torch.clamp(torch.round(v / scale), -127, 127).to(torch.int8)
            int8_state[k + ".q"] = q
            int8_state[k + ".scale"] = scale.reshape(1).to(torch.float32)
            scales[k] = float(scale)
        else:
            int8_state[k] = v
    save_file(int8_state, str(bundle / "model_int8.safetensors"), metadata={"format": "pt", "quant": "int8-symmetric-per-tensor"})
    mb8 = (bundle / "model_int8.safetensors").stat().st_size / 1e6

    # verify roundtrip max relative error on the biggest matrices
    errs = []
    for k, v in state.items():
        if k + ".q" in int8_state:
            dq = int8_state[k + ".q"].float() * int8_state[k + ".scale"]
            errs.append(float((dq - v).abs().max() / (v.abs().max() + 1e-9)))
    (bundle / "quant_report.json").write_text(json.dumps({
        "fp32": {"file": "model.safetensors", "mb": round((bundle / "model.safetensors").stat().st_size / 1e6, 2)},
        "fp16": {"file": "model_fp16.safetensors", "mb": round(mb16, 2)},
        "int8": {"file": "model_int8.safetensors", "mb": round(mb8, 2), "scheme": "per-tensor symmetric",
                 "max_rel_roundtrip_error": max(errs) if errs else 0.0,
                 "loader": "censer.load_state_dict_any(dir) dequantizes .q/.scale; or use --prefer-quant"},
    }, indent=2))
    print(f"fp32={(bundle/'model.safetensors').stat().st_size/1e6:.2f} MB  fp16={mb16:.2f} MB  int8={mb8:.2f} MB  max_rel_err={max(errs):.4f}")


if __name__ == "__main__":
    main()
