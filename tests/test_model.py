from __future__ import annotations

import torch

from kuclab.model import KucLabCneser, ModelConfig, estimate_params


def tiny_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=64, n_layers=2, n_heads=2, d_model=32, d_ff=64, max_seq_len=32)
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_shape():
    cfg = tiny_config()
    model = KucLabCneser(cfg)
    out = model(torch.randint(0, 64, (2, 16)))
    assert out.shape == (2, 16, 64)


def test_answer_logits_shape():
    cfg = tiny_config()
    model = KucLabCneser(cfg)
    logits = model.answer_logits(torch.randint(0, 64, (4, 12)))
    assert logits.shape == (4, 2)


def test_classify_range():
    cfg = tiny_config()
    model = KucLabCneser(cfg).eval()
    x = torch.randint(0, 64, (5, 10))
    probs = model.classify(x)
    assert probs.shape == (5, 2)  # [B, n_answers]
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    binary = model.classify_binary(x)
    assert binary.shape == (5,)
    assert torch.allclose(binary, probs[:, 1])


def test_classify_multi_label_head():
    cfg = tiny_config(answer_ids=(0, 1, 5, 6, 7))
    model = KucLabCneser(cfg).eval()
    probs = model.classify(torch.randint(0, 64, (4, 12)))
    assert probs.shape == (4, 5)
    logits = model.answer_logits(torch.randint(0, 64, (4, 12)))
    assert logits.shape == (4, 5)


def test_rope_scaling_linear_matches_hf_formula():
    # HF linear scaling: inv_freq /= factor  (transformers modeling_rope_utils)
    from kuclab.model import build_rope_cache
    import math
    cos_plain, _ = build_rope_cache(16, 8, torch.device("cpu"), torch.float32)
    cos_scaled, _ = build_rope_cache(16, 8, torch.device("cpu"), torch.float32,
                                     rope_scaling={"rope_type": "linear", "factor": 8.0})
    head_dim = 8
    base = 10000.0
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(16).float()
    # plain: angle = pos * inv ; scaled: angle = pos * inv / 8
    ang_plain = torch.outer(pos, inv)
    ang_scaled = torch.outer(pos, inv / 8.0)
    cos_ref = torch.cat((ang_plain.cos(), ang_plain.cos()), dim=-1)[:, None, :]
    cos_ref_s = torch.cat((ang_scaled.cos(), ang_scaled.cos()), dim=-1)[:, None, :]
    assert torch.allclose(cos_plain, cos_ref, atol=1e-5)
    assert torch.allclose(cos_scaled, cos_ref_s, atol=1e-5)


def test_tied_embeddings_share_storage():
    model = KucLabCneser(tiny_config(tie_embeddings=True))
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_padding_row_is_zero():
    model = KucLabCneser(tiny_config(pad_id=0))
    assert torch.all(model.embed.weight[0] == 0)


def test_causality_prefix_independence():
    torch.manual_seed(0)
    model = KucLabCneser(tiny_config()).eval()
    seq = torch.randint(1, 64, (1, 10))
    full = model(seq)
    prefix = model(seq[:, :6])
    assert torch.allclose(full[:, :6], prefix, atol=1e-5)


def test_variable_length_batches_work():
    model = KucLabCneser(tiny_config()).eval()
    long = torch.randint(1, 64, (2, 20))
    short = torch.randint(1, 64, (1, 5))
    assert model.answer_logits(long).shape == (2, 2)
    assert model.answer_logits(short).shape == (1, 2)


def test_estimate_params_reasonable():
    cfg = tiny_config()
    estimate = estimate_params(cfg)
    actual = KucLabCneser(cfg).num_parameters()
    assert abs(estimate - actual) / actual < 0.15


def test_answer_logits_are_differentiable():
    cfg = tiny_config()
    model = KucLabCneser(cfg)
    logits = model.answer_logits(torch.randint(0, 64, (3, 8)))
    assert logits.requires_grad
    logits.sum().backward()
    assert model.embed.weight.grad is not None
    assert torch.any(model.embed.weight.grad != 0)


def test_classify_does_not_track_grad():
    model = KucLabCneser(tiny_config())
    probs = model.classify(torch.randint(0, 64, (2, 6)))
    assert not probs.requires_grad


def test_padding_does_not_change_prediction():
    torch.manual_seed(0)
    model = KucLabCneser(tiny_config(pad_id=0)).eval()
    seq = torch.randint(1, 64, (1, 7))
    alone = model.answer_logits(seq)
    padded_short = torch.cat([seq, torch.zeros(1, 9, dtype=torch.long)], dim=1)
    padded_long = torch.cat([seq, torch.zeros(1, 20, dtype=torch.long)], dim=1)
    assert torch.allclose(alone, model.answer_logits(padded_short), atol=1e-4)
    assert torch.allclose(alone, model.answer_logits(padded_long), atol=1e-4)


def test_batched_rows_match_individual_rows():
    torch.manual_seed(0)
    model = KucLabCneser(tiny_config(pad_id=0)).eval()
    a = torch.randint(1, 64, (1, 5))
    b = torch.randint(1, 64, (1, 12))
    batch = torch.zeros(2, 12, dtype=torch.long)
    batch[0, :5] = a
    batch[1, :12] = b
    out = model.answer_logits(batch)
    assert torch.allclose(out[0], model.answer_logits(a)[0], atol=1e-4)
    assert torch.allclose(out[1], model.answer_logits(b)[0], atol=1e-4)


def test_last_token_index():
    model = KucLabCneser(tiny_config(pad_id=0))
    ids = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 0, 0, 0]])
    assert model.last_token_index(ids).tolist() == [2, 1]
