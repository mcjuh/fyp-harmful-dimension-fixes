"""Attention-mask-aware mean pooling over token embeddings, and [CLS] (token 0) pooling."""

import numpy as np
import torch


def mean_pool_tensor(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool one layer's token embeddings, ignoring padding tokens (stays on the tensors' device).

    hidden_states: (batch, seq_len, hidden_dim)
    attention_mask: (batch, seq_len)
    returns: (batch, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """`mean_pool_tensor` for CPU tensors, returned as a numpy array."""
    return mean_pool_tensor(hidden_states, attention_mask).numpy()


def cls_pool_tensor(hidden_states: torch.Tensor) -> torch.Tensor:
    """The first-token ([CLS]) vector of one layer: (batch, seq_len, hidden_dim) -> (batch, hidden_dim). Padding is on the right, so token 0 is always real."""
    return hidden_states[:, 0, :]
