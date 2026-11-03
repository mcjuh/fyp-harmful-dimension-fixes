"""Where everything lives. Scripts and library code import locations from here; none builds a `results/...` path itself.

Layout and the pipeline that fills it: README.md.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
LOGS = REPO_ROOT / "logs"

# ---- embedding caches (gitignored, ~40 GB; SHA-256 of the STS part in results/audit/cache_manifest.json)
CACHE = Path(os.environ["FYP_CACHE"]) if os.environ.get("FYP_CACHE") else RESULTS / "cache"   # location can be set with the FYP_CACHE environment variable. STS: {model}/{source}.emb.npy and sentences/{source}.json|.pairs.npz
CLS_FINAL_CACHE = CACHE / "cls"       # legacy final-layer classification cache, read only for an optional consistency check when present
CLS_ALL_LAYERS_CACHE = CACHE / "cls_all"   # classification suite, test capped at 5,000: every hidden state, 12 tasks; superseded by CLS_CACHE
CLS_CACHE = CACHE / "cls_full"       # classification suite: final layer only, FULL test splits, 12 tasks, float32 (n, 768)
RET_CACHE = CACHE / "ret"             # retrieval suite: every hidden state, NanoBEIR

# ---- result directories
AUDIT = RESULTS / "audit"             # cache audit/verification, overlap audit and its masks
OVERLAP_MASKS = AUDIT / "overlap_masks.npz"
STS_BASELINES = RESULTS / "sts" / "baselines"
STS_TRUNCATION = RESULTS / "sts" / "truncation_and_ranked_removal"
STS_DIAGNOSTICS = RESULTS / "sts" / "diagnostics"
SUITE = RESULTS / "retrieval_classification"   # retrieval/ and classification/ plus the l* tables and report.md
MRL_PAIR = RESULTS / "matryoshka"     # the Matryoshka pair (bert-mrl, bert-no-mrl): matryoshka/{sts,retrieval,classification}/
TRANSFORMS = RESULTS / "transformations"   # corrections before the cut (transformations.py): transformations/{sts,retrieval,classification}/
TIMING = RESULTS / "timing"          # timing of the primitives behind each cut (timing.py)

# folder of each task type inside the result directories above; code keeps the short keys "sts", "ret", "cls"
SUITE_DIR = {"sts": "sts", "ret": "retrieval", "cls": "classification"}
