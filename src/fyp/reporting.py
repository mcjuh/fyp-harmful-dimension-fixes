"""Markdown table helper shared by the report renderers."""

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    """GitHub-flavoured markdown table; floats are formatted with `fmt`, everything else with str()."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)
