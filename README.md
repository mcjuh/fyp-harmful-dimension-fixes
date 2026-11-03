# Dimension Removal and Geometry Corrections in Frozen Text Embeddings

This repository contains the code and result tables for a study of which dimensions of a frozen text embedding can be removed or corrected, and what each intervention costs in
downstream performance. We reproduce and extend the dimension-removal experiments of Takeshita et al. (2025), *Randomly Removing 50% of Dimensions in Text Embeddings has
Minimal Impact on Retrieval and Classification Tasks*, and add held-out ranking of dimensions, geometry corrections (centering, standardization, mean-bias correction,
all-but-the-top, whitening) and a timing benchmark. No encoder is trained or modified.

Author: Marcus Chee Jen An (mcjuh).

## Models and tasks

Eight 768-d encoders are evaluated at the final layer. `bert-raw`, `roberta-raw` and `distilbert-raw` use mean pooling over tokens, as do `sbert-nli` and the Matryoshka pair
`bert-mrl` and `bert-no-mrl`. `unsup-simcse` and `sup-simcse` are read at the [CLS] token they were trained on, the supervised model through its pooler layer. The tasks are seven
semantic textual similarity (STS) datasets (Spearman correlation between cosine similarity and the gold score), 13 NanoBEIR retrieval datasets (nDCG@10) and 12 MTEB classification
tasks (few-shot logistic regression probe, accuracy).

## Setup

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132   # or the build for your CUDA version, or CPU
pip install -r requirements.txt
pip install -e .
pytest tests -q
```

The extraction scripts write the embeddings to a cache under `results/cache/` (about 40 GB, created on the first run and not included in the repository). The cache location can be changed with the `FYP_CACHE` environment variable.
Checkpoint and dataset revisions are pinned in `src/fyp/models.py` and `src/fyp/corpora.py`.

## Pipeline

All commands run from the repository root. Every analysis script takes `--models` and `--out`. Development runs use `--split dev` with `--out`, so that the test splits are
not computed while the design is still being changed.

| Step | Script | Output |
|---|---|---|
| 1 | `extract_sts_embeddings.py` | STS and NLI-sample embeddings (cache) |
| 2 | `extract_retrieval_classification_embeddings.py` (`--cls-full` for the full classification test splits) | retrieval and classification embeddings (cache) |
| 3 | `check_cache.py`, `check_sentence_overlap.py` | `results/audit/` |
| 4 | `sts_baselines.py` | `results/sts/baselines/` (full-embedding scores, and a sanity check against the published SimCSE numbers) |
| 5 | `sts_truncation_and_ranked_removal.py` | `results/sts/truncation_and_ranked_removal/` |
| 6 | `sts_diagnostics.py` | `results/sts/diagnostics/` (reliability, shared dimensions, outliers, PCA) |
| 7 | `retrieval_classification_truncation_and_ranked_removal.py --suite ret\|cls --split test` | `results/retrieval_classification/{retrieval,classification}/` |
| 8 | `transformations.py --suite sts\|ret\|cls --split test` | `results/transformations/` |
| 9 | `timing_benchmark.py` | `results/timing/` |
| 10 | `report_sts.py`, `report_sts_diagnostics.py`, `report_retrieval_classification.py` (and `--transforms`), `report_matryoshka.py`, `figures_summary.py` | summary tables, reports and the three figures in `figures/` |

The Matryoshka pair is run with the same scripts, using `--models bert-mrl bert-no-mrl --out results/matryoshka/<sts|retrieval|classification>`.

## Terms used in the tables

| Term | Meaning |
|---|---|
| RP | relative performance: the score after the intervention divided by the score of the full embedding, macro-averaged over datasets |
| K, n | K is the percentage of dimensions removed (0 to 90 in steps of 5), and n is the number of dimensions removed |
| method `random`, `last`, `first` | remove a random subset, the last K% (keeping a prefix), or the first K% (keeping a suffix) |
| `in_sample` | dimensions ranked on the same data used for evaluation, as in Takeshita et al. (2025) |
| `held_out` | dimensions ranked on other data: for STS, the other test sets plus filtered STS-B training pairs; for retrieval and classification, the other datasets or tasks |
| `train_ranked` | STS only: dimensions ranked on the overlap-filtered STS-B training pairs |
| degrading / informative dimension | removing it raises / lowers the score of the evaluation task |
| `best_first`, `worst_first` | remove the dimensions with the highest / lowest leave-one-out score first |

## Result folders

| Folder | Content |
|---|---|
| `results/sts/` | baselines, truncation and ranked removal, diagnostics |
| `results/retrieval_classification/` | the same analyses for retrieval and classification |
| `results/matryoshka/` | the Matryoshka pair on all three task types, with its prediction checks |
| `results/transformations/` | geometry corrections: curves and a summary per task type |
| `results/timing/` | search and fit timings |
| `results/audit/` | sentence-overlap audit and masks, cache checksums |

Each table has a `.meta.json` sidecar with the script, seeds, library versions and device. The sidecars record the script names and commit hashes of the development
repository, which is not part of this repository.

## Hardware and timings

All runs, including the timing benchmark, were made on one desktop machine: an AMD Ryzen 7 7800X3D (8 cores, 16 threads), 32 GB of RAM, an NVIDIA GeForce RTX 5070 (12 GB) and
Windows 11, with Python 3.14, PyTorch 2.14 (CUDA 13.2) and TF32 disabled. The timings in `results/timing/` (search time against kept width, fit and projection costs) describe this
machine only. GPU timings were stable between repeated runs, and CPU timings varied between runs, so they should not be compared across machines.

## Notes

Cosine similarity is computed as `dot / sqrt(|a|² |b|²)` rather than `dot / (|a| |b|)`, so that identical sentences tie exactly. Ablation arithmetic is in float64, and the cached embeddings
are float32.

## Acknowledgements

Claude (Anthropic's AI assistant, used through Claude Code) assisted with writing and refactoring code, running the experiments and drafting documentation. The research
questions, design decisions and interpretation of the results are the author's, and all results were reviewed by the author.

## License

MIT, see `LICENSE`.
