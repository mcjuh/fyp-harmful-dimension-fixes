"""Dataset registry, with every Hub revision pinned.

Split roles:
  E   (evaluation, locked): STS12-16 test, STS-B test, SICK-R test
  R   (ranking):            STS-B train (overlap-filtered by results/audit/overlap_masks.npz, not here)
  AUX (ranking-set agreement only):         SICK-R train, unfiltered. 98.5% of its pairs share a sentence with
                            SICK-R test, so it is excluded from R and is
                            never used to rank dimensions that are then evaluated on E
  DEV (development + l*):   STS-B validation, SICK-R validation
  NLI (PCA corpus):        20k sentences sampled with a fixed seed from all-nli

SICK-R comes from `RobZamp/sick` (auto-converted parquet of the Zenodo SICK release), not
`mteb/sickr-sts`: the mteb copy puts all 9,927 pairs in a single "test" split, so it has no
train/validation split and its "test" contains the train portion.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from datasets import load_dataset


@dataclass(frozen=True)
class PairSource:
    key: str            # our name, e.g. "stsb_test"
    task: str           # task family, e.g. "stsb"
    role: str           # "E" | "R" | "AUX" | "DEV"
    hf_name: str
    revision: str
    split: str
    col1: str
    col2: str
    score_col: str


_MTEB_REV = {
    "sts12": "3a51344839ec620dd3b753e0945bdf6c7e37af3e",
    "sts13": "f475812ca75868066186343ebf60e01f95c75164",
    "sts14": "8ab5e540b20b58d42fa6b944d3accee9df3fbaca",
    "sts15": "ae8e1ecbd3b3bcf5c1c8fd798290033715d98b93",
    "sts16": "d8e3be175a943eecfea09c96795087771996197c",
    "stsb": "96943a16ea6a35129e253c659081cb59daf81b30",
}
SICK_HF = "RobZamp/sick"
SICK_REV = "3c0dde93388355f54b56e9f769e45d2b86bd0a6b"  # sha of refs/convert/parquet

NLI_HF = "sentence-transformers/all-nli"
NLI_CONFIG = "pair-class"
NLI_REV = "d482672c8e74ce18da116f430137434ba2e52fab"
NLI_SAMPLE_SIZE = 20_000
NLI_SAMPLE_SEED = 20251


def _mteb(task: str, split: str, role: str, key: str) -> PairSource:
    hf_name = "mteb/stsbenchmark-sts" if task == "stsb" else f"mteb/{task}-sts"
    return PairSource(key, task, role, hf_name, _MTEB_REV[task], split, "sentence1", "sentence2", "score")


def _sick(split: str, role: str, key: str) -> PairSource:
    return PairSource(key, "sickr", role, SICK_HF, SICK_REV, split, "sentence_A", "sentence_B", "relatedness_score")


PAIR_SOURCES: Dict[str, PairSource] = {s.key: s for s in [
    _mteb("sts12", "test", "E", "sts12_test"),
    _mteb("sts13", "test", "E", "sts13_test"),
    _mteb("sts14", "test", "E", "sts14_test"),
    _mteb("sts15", "test", "E", "sts15_test"),
    _mteb("sts16", "test", "E", "sts16_test"),
    _mteb("stsb", "test", "E", "stsb_test"),
    _sick("test", "E", "sickr_test"),
    _mteb("stsb", "train", "R", "stsb_train"),
    _sick("train", "AUX", "sickr_train"),
    _mteb("stsb", "validation", "DEV", "stsb_dev"),
    _sick("validation", "DEV", "sickr_dev"),
]}

E_KEYS = [k for k, s in PAIR_SOURCES.items() if s.role == "E"]
R_KEYS = [k for k, s in PAIR_SOURCES.items() if s.role == "R"]
AUX_KEYS = [k for k, s in PAIR_SOURCES.items() if s.role == "AUX"]
DEV_KEYS = [k for k, s in PAIR_SOURCES.items() if s.role == "DEV"]
NLI_KEY = "nli20k"


def load_pairs(key: str) -> dict:
    """Returns {'sentence1': [...], 'sentence2': [...], 'score': float64 array} for one pair source."""
    src = PAIR_SOURCES[key]
    ds = load_dataset(src.hf_name, split=src.split, revision=src.revision)
    return {
        "sentence1": list(ds[src.col1]),
        "sentence2": list(ds[src.col2]),
        "score": np.asarray(ds[src.score_col], dtype=np.float64),
    }


def load_nli_sample() -> List[str]:
    """20k unique sentences, sampled with a fixed seed from the unique premise+hypothesis
    sentences of all-nli (pair-class, train), in sorted order before sampling so the
    sample does not depend on set iteration order."""
    ds = load_dataset(NLI_HF, NLI_CONFIG, split="train", revision=NLI_REV)
    unique = sorted(set(ds["premise"]) | set(ds["hypothesis"]))
    rng = np.random.default_rng(NLI_SAMPLE_SEED)
    idx = np.sort(rng.choice(len(unique), size=NLI_SAMPLE_SIZE, replace=False))
    return [unique[i] for i in idx]


def normalize_sentence(s: str) -> str:
    """NFKC -> lowercase -> collapse whitespace -> strip. Used by the overlap audit
    and the cluster bootstrap to decide when two sentences are "the same"."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).lower()).strip()


def dedupe_pairs(sentence1: List[str], sentence2: List[str]):
    """Unique-sentence table + per-pair row indices (exact string match, first-appearance
    order over s1[0], s2[0], s1[1], s2[1], ...). Row mapping: pair i uses rows idx1[i], idx2[i]."""
    table: Dict[str, int] = {}
    idx1 = np.empty(len(sentence1), dtype=np.int64)
    idx2 = np.empty(len(sentence2), dtype=np.int64)
    for i, (a, b) in enumerate(zip(sentence1, sentence2)):
        idx1[i] = table.setdefault(a, len(table))
        idx2[i] = table.setdefault(b, len(table))
    return list(table), idx1, idx2
