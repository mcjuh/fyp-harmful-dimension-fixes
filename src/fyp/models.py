"""Frozen encoders for layer-wise sentence embeddings, pooled the way each model was trained.

Controlled extraction config (one config for every model):
  - revision-pinned HF checkpoints, fp32, eval mode, TF32 off (set by fyp.runtime.setup_device)
  - max_length 128 with truncation (truncated-sentence count is logged by the caller)
  - only the inputs a model actually uses are passed: DistilBERT gets no token_type_ids
    (its tokenizer returns them in transformers 5.x)
  - pooling per model (`ModelSpec.pooling`): attention-mask-aware mean of every hidden state (layer 0 = embedding output) for every
    model except the SimCSE pair, which is read at the [CLS] token (`cls`: token 0 of each hidden state; `cls_pooler`: the same, but
    the final slot is the checkpoint's pooler output, the SimCSE "cls" setting)
  - default batch policy: input order, batch size 128, no length sorting (see `encode` for the length-sorted option)
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from fyp.pooling import cls_pool_tensor, mean_pool, mean_pool_tensor

MAX_LENGTH = 128
BATCH_SIZE = 128


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_name: str
    revision: str
    num_hidden_states: int
    uses_token_type_ids: bool
    pooling: str = "mean"          # "mean" | "cls" | "cls_pooler"
    cache_name: str = ""           # cache directory name; "" = the key. The [CLS] caches sit beside the legacy mean-pooled ones

    @property
    def cache_dir_name(self) -> str:
        return self.cache_name or self.key

    @property
    def final_layer(self) -> int:
        """Index of the last hidden state (12; 6 for DistilBERT): the layer every ablation runs on."""
        return self.num_hidden_states - 1


MODEL_SPECS = {m.key: m for m in [
    ModelSpec("bert-raw", "bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594", 13, True),
    # SimCSE is trained on the [CLS] vector (supervised: through the pooler MLP; unsupervised: the MLP is dropped at test time).
    ModelSpec("sup-simcse", "princeton-nlp/sup-simcse-bert-base-uncased", "2d82fab19ac3a73a20dd20333d27eb8a52d6e97f", 13, True,
              pooling="cls_pooler", cache_name="sup-simcse-cls"),
    ModelSpec("unsup-simcse", "princeton-nlp/unsup-simcse-bert-base-uncased", "218baf6bc48c463a53f5b4e1c568299537fab124", 13, True,
              pooling="cls", cache_name="unsup-simcse-cls"),
    ModelSpec("sbert-nli", "sentence-transformers/bert-base-nli-mean-tokens", "160a52b38a51ae87295ec3eabcf11755e5d27a8d", 13, True),
    ModelSpec("roberta-raw", "roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b", 13, False),
    ModelSpec("distilbert-raw", "distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be", 7, False),
    # Matryoshka pair (Takeshita et al. 2026): bert-base-uncased, all-nli, multiple-negatives ranking loss, mean pooling;
    # the MRL arm adds MatryoshkaLoss over 768/512/256/128/64. Opt-in: not in MODEL_KEYS, so no default run includes them.
    ModelSpec("bert-mrl", "sobamchan/bert-base-uncased-mrl-768-512-256-128-64", "8df1048488339e0cc6cb278e8964fb1402972a05", 13, True),
    ModelSpec("bert-no-mrl", "sobamchan/bert-base-uncased-no-mrl", "f77fe9fa0fc94d8605b097cd2a1587631931f180", 13, True),
]}
MODEL_KEYS = ["bert-raw", "sup-simcse", "unsup-simcse", "sbert-nli", "roberta-raw", "distilbert-raw"]  # the six main models: every default
MRL_MODEL_KEYS = ["bert-mrl", "bert-no-mrl"]  # the Matryoshka pair, requested with --models
ALL_MODEL_KEYS = MODEL_KEYS + MRL_MODEL_KEYS


def final_layer(model: str) -> int:
    return MODEL_SPECS[model].final_layer


def cache_dir(root, model: str):
    """`root / <cache directory of the model>`; the only place a model key becomes a cache path."""
    return root / MODEL_SPECS[model].cache_dir_name


class LayerwiseEncoder:
    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, revision=spec.revision)
        self.model = AutoModel.from_pretrained(spec.hf_name, revision=spec.revision, dtype=torch.float32).to(device)
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers + 1
        self.hidden_size = self.model.config.hidden_size
        assert self.num_layers == spec.num_hidden_states, (spec.key, self.num_layers)
        assert self.hidden_size == 768, (spec.key, self.hidden_size)
        self.input_names = ["input_ids", "attention_mask"] + (["token_type_ids"] if spec.uses_token_type_ids else [])

    def count_truncated(self, sentences: List[str], max_length: int = MAX_LENGTH) -> int:
        lengths = [len(ids) for ids in self.tokenizer(sentences, truncation=False)["input_ids"]]
        return int(sum(n > max_length for n in lengths))

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor, layer: int, model_out) -> torch.Tensor:
        """(batch, 768) embedding of one hidden state under the model's pooling. The pooler output replaces the last slot only."""
        if self.spec.pooling == "mean":
            return mean_pool_tensor(hidden, attention_mask)
        if self.spec.pooling == "cls_pooler" and layer == self.num_layers - 1:
            return model_out.pooler_output.float()
        return cls_pool_tensor(hidden)

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH,
               sort_by_length: bool = False, pool_on_device: bool = False, layers=None) -> np.ndarray:
        """Returns float32 array of shape (len(sentences), num_layers, 768); with `layers` (hidden-state indices), only those
        layers are pooled and stored: (len(sentences), len(layers), 768).

        Defaults reproduce the controlled STS extraction config exactly (max_length 128, input order, CPU pooling).
        The retrieval extraction passes max_length=512, sort_by_length=True (length-sorted batches; padding
        is masked, so a sentence's embedding is unchanged up to fp32 noise) and pool_on_device=True (the same
        masked mean computed on the GPU, which avoids copying 13 x batch x 512 x 768 floats to the CPU)."""
        layers = list(range(self.num_layers)) if layers is None else list(layers)
        out = np.empty((len(sentences), len(layers), self.hidden_size), dtype=np.float32)
        if sort_by_length:
            lens = [len(ids) for ids in self.tokenizer(sentences, truncation=True, max_length=max_length)["input_ids"]]
            order = np.argsort(lens, kind="stable")
        else:
            order = np.arange(len(sentences))
        for start in range(0, len(sentences), batch_size):
            rows = order[start : start + batch_size]
            batch = [sentences[i] for i in rows]
            # sorted batches: pad to a multiple of 64 so only a handful of distinct shapes ever reach the CUDA allocator
            # (every new length otherwise reserves fresh memory, which spills past 12 GB and slows batches ~100x)
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                                 pad_to_multiple_of=64 if sort_by_length else None)
            inputs = {k: enc[k].to(self.device) for k in self.input_names}
            out_m = self.model(**inputs, output_hidden_states=True)
            hidden_states = out_m.hidden_states
            mask = enc["attention_mask"]
            if pool_on_device or self.spec.pooling != "mean":   # [CLS] pooling is a copy, so the device does not matter
                pooled = torch.stack([self._pool(hidden_states[l].float(), inputs["attention_mask"], l, out_m) for l in layers], dim=1)
                out[rows] = pooled.cpu().numpy()
            else:
                for j, l in enumerate(layers):
                    out[rows, j] = mean_pool(hidden_states[l].float().cpu(), mask)
            del hidden_states
            if sort_by_length and (start // batch_size) % 20 == 19 and self.device == "cuda":
                torch.cuda.empty_cache()
        return out


def load_encoder(key: str, device: str) -> LayerwiseEncoder:
    return LayerwiseEncoder(MODEL_SPECS[key], device)
