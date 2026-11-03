"""Loading cached STS pair data and computing float64 pair cosines / Spearman."""

import numpy as np
from scipy.stats import spearmanr

from fyp.models import cache_dir
from fyp.paths import CACHE


def load_pairs_index(source: str) -> dict:
    p = np.load(CACHE / "sentences" / f"{source}.pairs.npz")
    return {"idx1": p["idx1"], "idx2": p["idx2"], "score": p["score"].astype(np.float64)}


def load_embeddings(model: str, source: str, mmap: bool = True) -> np.ndarray:
    """float32 (n_unique_sentences, n_hidden_states, 768)."""
    return np.load(cache_dir(CACHE, model) / f"{source}.emb.npy", mmap_mode="r" if mmap else None)


def load_layer_pairs(model: str, source: str, layer: int):
    """(a, b, gold) for one layer: a, b are float64 (n_pairs, 768)."""
    emb = load_embeddings(model, source)
    idx = load_pairs_index(source)
    layer_emb = np.asarray(emb[:, layer, :], dtype=np.float64)
    return layer_emb[idx["idx1"]], layer_emb[idx["idx2"]], idx["score"]


def pair_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """dot / sqrt(|a|^2 |b|^2), never dot / (|a| |b|): identical-sentence pairs must tie exactly."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.einsum("nd,nd->n", a, b) / np.sqrt(np.einsum("nd,nd->n", a, a) * np.einsum("nd,nd->n", b, b))


def spearman(pred: np.ndarray, gold: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties (scipy default)."""
    return float(spearmanr(pred, gold).statistic)
