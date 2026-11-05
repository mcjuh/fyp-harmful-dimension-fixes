"""Protocol constants shared by every analysis: dimensions, K grid, seeds, thresholds.

One definition each. Scripts import these instead of re-declaring them.
"""

N_DIMS = 768

# model groups: the paper's setting is the contrastive group; the raw models are exploratory
CONTRASTIVE_MODELS = ["sup-simcse", "unsup-simcse", "sbert-nli", "bert-mrl", "bert-no-mrl"]  # the last two: the Matryoshka pair, trained with in-batch-negative contrastive loss
RAW_MODELS = ["bert-raw", "roberta-raw", "distilbert-raw"]


def n_remove(k_percent: float) -> int:
    """Number of coordinates deleted at K%: the complement of the authors' keep count `int(dimension * resize_scale)`
    (`Truncator.encode`, sobamchan/random-embedding-truncation commit 780eb94) with resize_scale = (100 - K) / 100."""
    return N_DIMS - int(N_DIMS * ((100 - k_percent) / 100))


K_GRID = list(range(0, 95, 5))          # K = 0, 5, ..., 90 (%)
N_GRID = [n_remove(k) for k in K_GRID]
N25, N50 = n_remove(25), n_remove(50)
RANKING_POWER_K = [10, 15, 20, 25, 30, 35, 40, 45, 50]    # K range on which "best-first above / worst-first below the random band" is judged
ROBUSTNESS_THRESHOLD = 0.90                    # random 50% removal must keep at least this relative performance

# ---- seeds
SEED_TIE = 7                           # tie-break permutation of every ranking
SEEDS_RANDOM_K = range(0, 100)         # random-K truncation (STS; the retrieval suite uses the same 100)
SEEDS_DEGRADING_RANDOM = range(1000, 1100)     # random orders within the degrading dimensions 
SEEDS_INFORMATIVE_RANDOM = range(2000, 2100)     # random orders within the informative dimensions 
SEED_AUTOCORR, N_AUTOCORR_PERM = 3000, 1000
SEEDS_SPLIT_HALF = range(4000, 4100)   # split-half reliability
SEED_DELTA_BOOT = 5000                 # bootstrap of per-dimension scores; task i of [ranking set, evaluation sets...] uses 5000 + i
SEEDS_ROTATION = range(6000, 6020)     # rotation null
SEED_SHARED, N_SHARED_NULL = 7000, 1000
SEEDS_OUTLIER = range(8000, 8010)
Z_OUTLIER = 3.0
SEED_BOOT, N_BOOT = 9000, 1000         # headline paired bootstrap; task i uses 9000 + i
SEED_BASE_SAMPLE = 1                   # cache verification: the 200 random sentences

# ---- classification few-shot protocol
N_FID_RANDOM = 10                      # random-K seeds that also get the label-free fidelity measure (the paper ran ten)


# ---- the authors' task lists, copied verbatim
# Source: sobamchan/random-embedding-truncation, src/random_embedding_truncation/utils.py, commit 780eb94.
def convert_dataset_name_for_nanobeir(dataset_name: str) -> str:
    if dataset_name == "climate-fever":
        dataset_name = "climatefever"
    if dataset_name == "dbpedia-entity":
        dataset_name = "dbpedia"
    if dataset_name == "fiqa":
        dataset_name = "fiqa2018"
    if dataset_name == "quora":
        dataset_name = "quoraretrieval"
    if dataset_name == "webis-touche2020":
        dataset_name = "touche2020"
    return dataset_name


def get_nanobeir_task_ids() -> list[str]:
    return [
        "NanoBEIR",
        "NanoArguAna",
        "NanoClimateFEVER",
        "NanoDBPedia",
        "NanoFEVER",
        "NanoFiQA2018",
        "NanoHotpotQA",
        "NanoMSMARCO",
        "NanoNFCorpus",
        "NanoNQ",
        "NanoQuoraRetrieval",
        "NanoSCIDOCS",
        "NanoSciFact",
        "NanoTouche2020",
    ]


def get_mteb_task_ids() -> list[str]:
    return [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ]


def get_beir_task_ids() -> list[str]:
    return [
        "arguana",
        "climate-fever",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        "msmarco",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
        "webis-touche2020",
    ]
