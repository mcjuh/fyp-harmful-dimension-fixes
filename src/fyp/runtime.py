"""Device selection, numerics settings, and per-result-file run metadata.

Every script that produces a result calls `setup_device()` once and passes the returned
dict to `run_metadata()` when writing outputs (git commit, library
versions, device actually used, seeds, protocol id, timestamp on every result file).
"""

import argparse
import datetime as _dt
import importlib.metadata
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch

from fyp.paths import REPO_ROOT

_LIBS = ("torch", "transformers", "sentence-transformers", "numpy", "scipy", "scikit-learn", "datasets")


def add_device_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--require-gpu", action="store_true", help="abort instead of falling back to CPU")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto = GPU if available, else CPU with a warning")


def setup_device(require_gpu: bool = False, device: str = "auto") -> dict:
    """Pick the device, set fp32 numerics (TF32 off, matmul precision 'highest'), and
    return a description dict that goes into run metadata. Also makes stdout UTF-8 (reports print rho, >=, etc.)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    cuda_ok = torch.cuda.is_available()
    fallback_reason = None
    if device == "cpu":
        chosen = "cpu"
        fallback_reason = "CPU explicitly requested via --device cpu"
    elif cuda_ok:
        chosen = "cuda"
    else:
        reason = f"torch.cuda.is_available() is False (torch {torch.__version__}, cuda build {torch.version.cuda})"
        if require_gpu or device == "cuda":
            sys.exit(f"ERROR: GPU required but unavailable: {reason}")
        chosen = "cpu"
        fallback_reason = reason
        warnings.warn(f"\n\n*** WARNING: FALLING BACK TO CPU: {reason} ***\n")

    info = {
        "device": chosen,
        "device_name": torch.cuda.get_device_name(0) if chosen == "cuda" else platform.processor(),
        "cpu_fallback_reason": fallback_reason,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    note = ""
    if fallback_reason and device == "cpu":
        note = "  (CPU explicitly requested)"
    elif fallback_reason:
        note = f"  <-- CPU FALLBACK: {fallback_reason}"
    print(f"[fyp] device in use: {info['device']} ({info['device_name']}){note}")
    return info


def git_state() -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    # A run's own fresh outputs (untracked files under output dirs) don't make the code dirty;
    # modified tracked files, or untracked files anywhere else (e.g. a new script), do.
    output_dirs = ("results/", "logs/", "figures/")
    dirty_lines = [] if status is None else [
        ln for ln in status.splitlines()
        if not (ln.startswith("?? ") and ln[3:].strip('"').startswith(output_dirs))
    ]
    return {
        "git_commit": commit,  # None => repository has no commits yet
        "git_dirty": bool(dirty_lines) if status is not None else None,
        "git_dirty_paths": dirty_lines[:20],
    }


def library_versions() -> dict:
    out = {"python": platform.python_version(), "torch_cuda": torch.version.cuda}
    for lib in _LIBS:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            out[lib] = None
    return out


def run_metadata(device_info: dict, *, script: str, protocol_id: Optional[str], seeds: Optional[dict] = None,
                 extra: Optional[dict] = None) -> dict:
    return {
        "script": script,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol_id": protocol_id,
        "seeds": seeds or {},
        **git_state(),
        "libraries": library_versions(),
        **device_info,
        **(extra or {}),
    }


def write_metadata(path: Path, meta: dict) -> None:
    """Sidecar metadata file: `<result>.meta.json` next to `<result>`."""
    path = Path(path)
    sidecar = path.with_name(path.name + ".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, default=str))


def save_table(df, path: Path, meta: dict) -> None:
    """Write a result table as CSV (`.csv.gz` is compressed by pandas) plus its metadata sidecar."""
    df.to_csv(path, index=False)
    write_metadata(path, meta)
