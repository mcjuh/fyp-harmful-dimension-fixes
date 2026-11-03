"""Model registry (the Matryoshka pair is opt-in) and the stage timer."""

import time

from fyp.models import ALL_MODEL_KEYS, MODEL_KEYS, MODEL_SPECS, MRL_MODEL_KEYS
from fyp.protocol import CONTRASTIVE_MODELS
from fyp.timing import StageTimer


def test_default_models_unchanged_and_pair_is_opt_in():
    assert MODEL_KEYS == ["bert-raw", "sup-simcse", "unsup-simcse", "sbert-nli", "roberta-raw", "distilbert-raw"]
    assert MRL_MODEL_KEYS == ["bert-mrl", "bert-no-mrl"]
    assert ALL_MODEL_KEYS == MODEL_KEYS + MRL_MODEL_KEYS
    assert set(MODEL_SPECS) == set(ALL_MODEL_KEYS)
    for k in MRL_MODEL_KEYS:
        spec = MODEL_SPECS[k]
        assert len(spec.revision) == 40 and spec.num_hidden_states == 13 and spec.final_layer == 12
        assert k in CONTRASTIVE_MODELS


def test_stage_timer_marks_and_stages():
    t = StageTimer(model="m", suite="s")
    time.sleep(0.02)
    t.mark("a", "load")
    with t.stage("b", "one_time", n_items=3):
        time.sleep(0.02)
    time.sleep(0.02)
    t.mark("c", "bookkeeping")
    df = t.table()
    assert list(df["stage"]) == ["a", "b", "c"] and list(df["role"]) == ["load", "one_time", "bookkeeping"]
    assert (df["seconds"] >= 0.015).all()
    assert df["seconds"].iloc[2] < 0.04          # time spent inside the "b" block does not leak into the next mark
    assert (df["model"] == "m").all() and df["n_items"].iloc[1] == 3
