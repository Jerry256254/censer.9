from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import Batcher, SafetyDataset, collate

CURATED_SOURCES = {"hard_negative", "borderline", "curated_unsafe", "synthetic_safe", "synthetic_unsafe", "aug_scraped", "adversarial", "must_catch"}
SAFE_CURATED_SOURCES = {"hard_negative", "borderline", "synthetic_safe"}


def _is_curated(source: str) -> bool:
    return source in CURATED_SOURCES or source.startswith("synthetic_ml_unsafe") or source.startswith("synthetic_ml_safe")


def _is_safe_curated(source: str) -> bool:
    return source in SAFE_CURATED_SOURCES or source.startswith("synthetic_ml_safe")
from .model import KucLabCneser, ModelConfig
from .tokenizer import KucTokenizer, train_tokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_at(step: int, total: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def ensure_tokenizer(config: dict, data_dir: Path, out_dir: Path) -> KucTokenizer:
    if (out_dir / "tokenizer.json").exists():
        return KucTokenizer.load(out_dir)
    shared = Path("checkpoints/tokenizer")
    if (shared / "tokenizer.json").exists():
        import shutil

        shutil.copytree(shared, out_dir, dirs_exist_ok=True)
        return KucTokenizer.load(out_dir)

    texts: list[str] = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                texts.append(json.loads(line)["text"])
    limit = config["tokenizer"]["corpus_samples"]
    texts = texts[:limit]
    print(f"training tokenizer on {len(texts)} samples ...")
    return train_tokenizer(
        texts,
        vocab_size=config["tokenizer"]["vocab_size"],
        min_frequency=config["tokenizer"]["min_frequency"],
        out_dir=out_dir,
    )


@torch.no_grad()
def evaluate(model: KucLabCneser, loader: Batcher, pad_id: int, device: torch.device) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    batches = 0
    for batch in loader:
        input_ids, answer_ids, labels = collate(batch, pad_id)
        input_ids = input_ids.to(device)
        answer_ids = answer_ids.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.answer_logits(input_ids)
        bin_logits = logits[:, :2] if logits.shape[1] > 2 else logits
        loss = F.cross_entropy(bin_logits, labels)
        total_loss += loss.item()
        batches += 1
        preds = bin_logits.argmax(-1)
        tp += int(((preds == 1) & (labels == 1)).sum())
        fp += int(((preds == 1) & (labels == 0)).sum())
        tn += int(((preds == 0) & (labels == 0)).sum())
        fn += int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": total_loss / max(1, batches),
        "accuracy": (tp + tn) / max(1, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train(config: dict, data_dir: Path, out_dir: Path, resume: bool, init_checkpoint: str | None = None) -> None:
    torch.manual_seed(config["seed"])
    device = pick_device()
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    tok = ensure_tokenizer(config, data_dir, out_dir / "tokenizer")
    pad_id = tok.pad_id
    max_chars = config["max_chars"]
    max_len = int(config.get("train_seq_len", config["max_seq_len"]))
    infer_len = int(config["max_seq_len"])

    mcfg = ModelConfig(
        vocab_size=tok.vocab_size,
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_model=config["model"]["d_model"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=infer_len,
        dropout=config["model"]["dropout"],
        tie_embeddings=config["model"]["tie_embeddings"],
        pad_id=pad_id,
        answer_ids=tuple(tok.answer_ids),
        rope_scaling=config["model"].get("rope_scaling"),
    )
    model = KucLabCneser(mcfg).to(device)
    print(f"parameters: {model.num_parameters():,}")

    tcfg = config["train"]
    dcfg = tcfg.get("distill") or {}
    soft_path = Path(dcfg.get("soft_file", "")) if dcfg.get("soft_file") else None
    soft = None
    if soft_path is not None and soft_path.exists() and float(dcfg.get("alpha", 1.0)) < 1.0:
        import numpy as np

        soft = np.load(soft_path).tolist()
        print(f"distillation: loaded {len(soft)} soft labels from {soft_path} "
              f"(alpha={dcfg.get('alpha')}, T={dcfg.get('temperature')})")
    train_ds = SafetyDataset(data_dir / "train.jsonl", tok, max_len, augment_typo=float(tcfg.get("augment_typo", 0.0)), seed=config["seed"], soft=soft, augment_adversarial=float(tcfg.get("augment_adversarial", 0.0)))
    val_ds = SafetyDataset(data_dir / "val.jsonl", tok, max_len)
    max_tokens = tcfg["batch_size"] * max_len

    train_loader = Batcher(train_ds, tcfg["batch_size"], max_tokens, shuffle=True, seed=config["seed"])
    val_loader = Batcher(val_ds, tcfg["batch_size"] * 2, max_tokens * 2, shuffle=False, seed=config["seed"])

    epochs = tcfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch // max(1, tcfg["grad_accum_steps"])
    print(f"train={len(train_ds)} val={len(val_ds)} steps_per_epoch={steps_per_epoch} total={total_steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=tcfg["weight_decay"],
    )

    start_epoch = 0
    global_step = 0
    best_f1 = -1.0
    ckpt_path = out_dir / "last.pt"
    if init_checkpoint:
        st = torch.load(init_checkpoint, map_location=device, weights_only=False)
        from .model import remap_legacy_state
        st = remap_legacy_state(st)
        model.load_state_dict(st["model"], strict=True)
        print(f"initialized model weights from {init_checkpoint}")
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        from .model import remap_legacy_state
        state = remap_legacy_state(state)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        global_step = state["step"]
        best_f1 = state.get("best_f1", -1.0)
        print(f"resumed from {ckpt_path} (epoch {start_epoch})")

    log_path = out_dir / "train_log.jsonl"
    use_amp = tcfg["amp"] and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        running = 0.0
        seen = 0
        for i, batch in enumerate(train_loader):
            input_ids, answer_ids, labels = collate(batch, pad_id)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = lr_at(global_step, total_steps, tcfg["lr"], tcfg["min_lr"], tcfg["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            curated_weight = float(tcfg.get("curated_weight", 1.0))
            safe_curated_weight = float(tcfg.get("safe_curated_weight", curated_weight))
            unsafe_weight = float(tcfg.get("unsafe_weight", 1.0))
            must_catch_weight = float(tcfg.get("must_catch_weight", curated_weight))
            hard_negative_weight = float(tcfg.get("hard_negative_weight", safe_curated_weight))

            def _row_weight(item) -> float:
                src = item["source"]
                if src == "must_catch":
                    return must_catch_weight
                if src == "hard_negative":
                    return hard_negative_weight
                if _is_safe_curated(src):
                    return safe_curated_weight
                if _is_curated(src):
                    return curated_weight
                if item["label"] == 1:
                    return unsafe_weight
                return 1.0

            weights = torch.tensor(
                [_row_weight(item) for item in batch],
                dtype=torch.float32, device=device,
            )
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                logits = model.answer_logits(input_ids)  # [B, n_answers]; 0/1 = safe/unsafe
                # clamp + fp32 for the loss: raw logits can reach 1e4, and bf16
                # log_softmax at that magnitude loses all precision (KL went
                # negative and poisoned gradients)
                bin_logits = logits[:, :2].float().clamp(-30.0, 30.0)
                ce = F.cross_entropy(bin_logits, labels, reduction="none")
                alpha = float(dcfg.get("alpha", 1.0))
                temp = float(dcfg.get("temperature", 1.0))
                if alpha < 1.0 and soft is not None and temp > 1.0:
                    soft_t = torch.tensor([item["soft"] for item in batch], dtype=torch.float32, device=device)
                    pt = soft_t.clamp(1e-6, 1.0 - 1e-6)
                    boost = float(dcfg.get("curated_boost", 0.0))
                    if boost > 0.0:
                        is_curated = torch.tensor(
                            [_is_curated(item["source"]) for item in batch],
                            dtype=torch.float32, device=device,
                        )
                        pt = torch.where(
                            is_curated.bool(),
                            (1.0 - boost) * pt + boost * labels.float(),
                            pt,
                        ).clamp(1e-6, 1.0 - 1e-6)
                    # multilingual/adversarial rows: the English teachers score them
                    # ~safe, which contradicts the hard label and poisons the KL term.
                    # Use the hard label as the teacher target for those rows.
                    is_new_ml = torch.tensor(
                        [str(item["source"]).startswith(("synthetic_ml", "adversarial", "ml_scraped"))
                         for item in batch],
                        dtype=torch.bool, device=device,
                    )
                    pt = torch.where(is_new_ml, labels.float(), pt).clamp(1e-6, 1.0 - 1e-6)
                    log_student = F.log_softmax(bin_logits, dim=-1)  # bin_logits already fp32+clamped
                    kl = (pt * (torch.log(pt) - log_student[:, 1])
                          + (1.0 - pt) * (torch.log(1.0 - pt) - log_student[:, 0])) * (temp * temp)
                    per_sample = alpha * ce + (1.0 - alpha) * kl
                else:
                    per_sample = ce
                loss = (per_sample * weights).mean() / tcfg["grad_accum_steps"]
                # multi-label category head: BCE on logits[:, 2:] where cats are known
                if logits.shape[1] > 2:
                    cats = torch.tensor([item["cats"] for item in batch], dtype=torch.float32, device=device)
                    cats_mask = torch.tensor([item["cats_mask"] for item in batch], dtype=torch.float32, device=device)
                    cat_logits = logits[:, 2:].float().clamp(-30.0, 30.0)
                    bce = F.binary_cross_entropy_with_logits(cat_logits, cats, reduction="none") * cats_mask
                    cat_loss = (bce.sum(-1) / cats_mask.sum(-1).clamp(min=1.0)) * weights
                    loss = loss + float(tcfg.get("cat_weight", 1.0)) * cat_loss.mean() / tcfg["grad_accum_steps"]

            loss.backward()
            running += loss.item() * tcfg["grad_accum_steps"]
            seen += 1

            if (i + 1) % tcfg["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % tcfg["eval_every_steps"] == 0:
                    metrics = evaluate(model, val_loader, pad_id, device)
                    entry = {"step": global_step, "epoch": epoch, "lr": lr, **metrics}
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry) + "\n")
                    print(
                        f"  step {global_step:5d} loss={metrics['loss']:.4f} "
                        f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                        f"p={metrics['precision']:.4f} r={metrics['recall']:.4f}"
                    )
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        save_checkpoint(out_dir / "best.pt", model, tok, mcfg, metrics, global_step, max_chars=max_chars, name=config.get("name", "Censer.9"))
                    model.train()

        epoch_loss = running / max(1, seen)
        metrics = evaluate(model, val_loader, pad_id, device)
        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f} "
            f"val_f1={metrics['f1']:.4f} val_acc={metrics['accuracy']:.4f} ({elapsed:.0f}s)"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": global_step, "epoch": epoch, "epoch_loss": epoch_loss, **metrics}) + "\n")

        save_checkpoint(out_dir / f"epoch{epoch + 1}.pt", model, tok, mcfg, metrics, global_step, epoch=epoch, best_f1=best_f1, max_chars=max_chars, name=config.get("name", "Censer.9"))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(out_dir / "best.pt", model, tok, mcfg, metrics, global_step, max_chars=max_chars, name=config.get("name", "Censer.9"))

        save_checkpoint(ckpt_path, model, tok, mcfg, metrics, global_step, epoch=epoch, best_f1=best_f1, optimizer=optimizer, max_chars=max_chars, name=config.get("name", "Censer.9"))

    save_checkpoint(out_dir / "final.pt", model, tok, mcfg, metrics, global_step, max_chars=max_chars, name=config.get("name", "Censer.9"))
    print(f"done. best val f1={best_f1:.4f}")


def save_checkpoint(
    path: Path,
    model: KucLabCneser,
    tok: KucTokenizer,
    mcfg: ModelConfig,
    metrics: dict,
    step: int,
    epoch: int = 0,
    best_f1: float = -1.0,
    optimizer: torch.optim.Optimizer | None = None,
    max_chars: int = 500,
    name: str = "Censer.9",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_config": mcfg.__dict__,
        "metrics": metrics,
        "step": step,
        "epoch": epoch,
        "best_f1": best_f1,
        "max_chars": max_chars,
        "name": name,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Censer .4 from scratch.")
    parser.add_argument("--config", default="configs/censer9a.json")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", default=None, help="load model weights only (no optimizer) from this checkpoint at start")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    train(config, Path(args.data), Path(args.out), args.resume, init_checkpoint=args.init_checkpoint)


if __name__ == "__main__":
    main()
