"""Wall-clock timing of the stages of a result-producing script.

Every stage is tagged with a `role` so that cost can be separated the way a deployment would see it:
  - "one_time":    done once per model (or per task) to choose or fit a cut: leave-one-out attribution, a ranking, a PCA fit
  - "bookkeeping": research evaluation only (curves over the K grid, random-K seeds, bootstraps, nulls); not a deployment cost
  - "load":        reading cached embeddings
CUDA work is synchronised before each reading, so GPU time is counted in the stage that launched it.
Timings are not comparable across machines; the device is recorded in the run metadata of the table that holds them.
"""

import contextlib
import time

import pandas as pd
import torch

ROLES = ("one_time", "bookkeeping", "load")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    def __init__(self, **context):
        self.context = context
        self.rows = []
        _sync()
        self._last = time.perf_counter()

    def mark(self, stage: str, role: str, **extra) -> float:
        """Record the seconds since the previous mark (or since construction) as `stage`."""
        assert role in ROLES, role
        _sync()
        now = time.perf_counter()
        self.rows.append({**self.context, "stage": stage, "role": role, "seconds": now - self._last, **extra})
        self._last = now
        return self.rows[-1]["seconds"]

    @contextlib.contextmanager
    def stage(self, stage: str, role: str, **extra):
        """Time one block; the time spent inside does not leak into the next `mark`."""
        assert role in ROLES, role
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            t1 = time.perf_counter()
            self.rows.append({**self.context, "stage": stage, "role": role, "seconds": t1 - t0, **extra})
            self._last = t1

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
