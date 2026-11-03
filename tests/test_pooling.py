"""Pooling per model: [CLS] copies token 0, the pooler output replaces only the last slot, mean pooling is unchanged."""

from types import SimpleNamespace

import torch

from fyp.models import MODEL_SPECS, LayerwiseEncoder, cache_dir
from fyp.pooling import cls_pool_tensor, mean_pool_tensor
from pathlib import Path


def _encoder(key: str) -> LayerwiseEncoder:
    enc = object.__new__(LayerwiseEncoder)          # no checkpoint download: only `_pool` is under test
    enc.spec, enc.num_layers = MODEL_SPECS[key], 13
    return enc


def test_cls_pool_takes_token_zero():
    h = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    assert torch.equal(cls_pool_tensor(h), h[:, 0, :])


def test_pooling_per_model_and_slot():
    h, mask = torch.randn(3, 7, 768), torch.ones(3, 7)
    out = SimpleNamespace(pooler_output=torch.randn(3, 768))
    sup, unsup, bert = _encoder("sup-simcse"), _encoder("unsup-simcse"), _encoder("bert-raw")
    assert MODEL_SPECS["sup-simcse"].pooling == "cls_pooler" and MODEL_SPECS["unsup-simcse"].pooling == "cls"
    assert torch.equal(sup._pool(h, mask, 5, out), h[:, 0])                 # inner layers: plain [CLS]
    assert torch.equal(sup._pool(h, mask, 12, out), out.pooler_output)      # last slot: pooler output
    assert torch.equal(unsup._pool(h, mask, 12, out), h[:, 0])              # no pooler for the unsupervised model
    assert torch.equal(bert._pool(h, mask, 12, out), mean_pool_tensor(h, mask))


def test_cache_directories_leave_legacy_caches_alone():
    root = Path("x")
    assert cache_dir(root, "sup-simcse") == root / "sup-simcse-cls" and cache_dir(root, "unsup-simcse") == root / "unsup-simcse-cls"
    assert cache_dir(root, "bert-raw") == root / "bert-raw" and cache_dir(root, "sbert-nli") == root / "sbert-nli"
