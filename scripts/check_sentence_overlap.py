"""Overlap audit: exact-match overlap on normalized sentences between every ranking/eval set, then the filters on R and held_out.

Normalization: NFKC -> lowercase -> collapse whitespace -> strip.
Sentence-level overlap A vs B: unique normalized sentences of A that also occur in B, and the
number of A's pairs with at least one sentence in B.
Pair-level overlap A vs B: A's pairs whose *unordered* normalized pair (s1, s2) also occurs as a
pair in B.

Filters (never touching E itself):
  R keep mask (train_ranked):   drop an R pair if either sentence occurs in ANY E task.
  held_out fold masks:      for held-out task t, pairs of each other E task u are kept only if neither
                      sentence occurs in t (R is already filtered against all of E).
Outputs: results/audit/overlap_audit.md, overlap_audit.json, overlap_masks.npz (+ .meta.json)

Usage: python scripts/check_sentence_overlap.py [--out DIR]    (--out writes the three outputs to DIR instead of results/audit)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fyp.corpora import AUX_KEYS, DEV_KEYS, E_KEYS, NLI_KEY, R_KEYS, normalize_sentence
from fyp.paths import AUDIT, CACHE
from fyp.runtime import run_metadata, write_metadata

SENT = CACHE / "sentences"
OUT = AUDIT
PROTOCOL_ID = "sentence_overlap"
MAX_R_REMOVAL = 0.5     # the audit aborts if the filter would remove more than half of R


def load(key):
    table = [normalize_sentence(s) for s in json.loads((SENT / f"{key}.json").read_text())["sentences"]]
    if key == NLI_KEY:
        return {"sents": table, "pairs": None}
    p = np.load(SENT / f"{key}.pairs.npz")
    a = [table[i] for i in p["idx1"]]
    b = [table[i] for i in p["idx2"]]
    return {"sents": table, "a": a, "b": b, "pairs": {frozenset((x, y)) for x, y in zip(a, b)}}


def main():
    global OUT
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=AUDIT, help="output directory (default results/audit)")
    OUT = ap.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    keys = E_KEYS + R_KEYS + AUX_KEYS + DEV_KEYS + [NLI_KEY]
    data = {k: load(k) for k in keys}
    sentset = {k: set(d["sents"]) for k, d in data.items()}

    def compare(ka, kb):
        A, sb = data[ka], sentset[kb]
        inter = sentset[ka] & sb
        row = {"a": ka, "b": kb, "a_unique_sents": len(sentset[ka]), "shared_sents": len(inter),
               "shared_sents_frac_of_a": len(inter) / len(sentset[ka])}
        if A["pairs"] is not None:
            touch = sum((x in sb) or (y in sb) for x, y in zip(A["a"], A["b"]))
            row.update({"a_pairs": len(A["a"]), "a_pairs_touching_b": touch, "a_pairs_touching_b_frac": touch / len(A["a"])})
            if data[kb]["pairs"] is not None:
                pm = sum(frozenset((x, y)) in data[kb]["pairs"] for x, y in zip(A["a"], A["b"]))
                row.update({"a_pairs_exact_match_in_b": pm, "a_pairs_exact_match_frac": pm / len(A["a"])})
        return row

    rows = []
    for ka in R_KEYS + AUX_KEYS + DEV_KEYS + [NLI_KEY]:
        for kb in E_KEYS:
            rows.append(compare(ka, kb))
    for ka in E_KEYS:
        for kb in E_KEYS:
            if ka != kb:
                rows.append(compare(ka, kb))
    rows.append(compare("stsb_train", "sickr_train"))
    rows.append(compare("sickr_train", "stsb_train"))

    # ---- filters ----
    # AUX (SICK-R train) is filtered here only to document why it is not in R; its
    # mask is not saved and it is never used for ranking.
    e_all = set().union(*(sentset[k] for k in E_KEYS))
    masks, r_summary, aux_summary = {}, {}, {}
    for k in AUX_KEYS:
        d = data[k]
        keep = np.array([(x not in e_all) and (y not in e_all) for x, y in zip(d["a"], d["b"])])
        aux_summary[k] = {"pairs": len(keep), "kept": int(keep.sum()), "removed": int((~keep).sum()),
                          "removed_frac": float((~keep).mean())}
    for k in R_KEYS:
        d = data[k]
        keep = np.array([(x not in e_all) and (y not in e_all) for x, y in zip(d["a"], d["b"])])
        masks[f"R__{k}"] = keep
        r_summary[k] = {"pairs": len(keep), "kept": int(keep.sum()), "removed": int((~keep).sum()),
                        "removed_frac": float((~keep).mean())}
    r_total = sum(v["pairs"] for v in r_summary.values())
    r_removed = sum(v["removed"] for v in r_summary.values())
    r_summary["_total"] = {"pairs": r_total, "removed": r_removed, "removed_frac": r_removed / r_total}

    p2_summary = {}
    for t in E_KEYS:
        for u in E_KEYS:
            if u == t:
                continue
            d = data[u]
            keep = np.array([(x not in sentset[t]) and (y not in sentset[t]) for x, y in zip(d["a"], d["b"])])
            masks[f"held_out__{t}__{u}"] = keep
            p2_summary[f"{t}<-{u}"] = {"pairs": len(keep), "kept": int(keep.sum()), "removed": int((~keep).sum())}
        fold_pairs = sum(p2_summary[f"{t}<-{u}"]["kept"] for u in E_KEYS if u != t)
        p2_summary[f"_fold_{t}_ranking_pairs"] = {"E_minus_t_kept": fold_pairs,
                                                  "R_kept": sum(r_summary[k]["kept"] for k in R_KEYS)}

    np.savez(OUT / "overlap_masks.npz", **masks)
    report = {"normalization": "NFKC, lowercase, collapse whitespace, strip", "rows": rows,
              "R_filter": r_summary, "AUX_filter_not_applied": aux_summary, "held_out_filter": p2_summary}
    (OUT / "overlap_audit.json").write_text(json.dumps(report, indent=2))
    meta = run_metadata({"device": "cpu (no model computation)"}, script="scripts/check_sentence_overlap.py",
                        protocol_id=PROTOCOL_ID)
    write_metadata(OUT / "overlap_audit.json", meta)
    write_metadata(OUT / "overlap_masks.npz", meta)

    # ---- markdown ----
    L = ["# Overlap audit", "",
         f"Generated by `scripts/check_sentence_overlap.py` (git `{meta['git_commit']}`, dirty={meta['git_dirty']}). "
         "Normalization: NFKC → lowercase → collapse whitespace → strip. Exact match only; near-duplicates are not detected.", "",
         "Columns: **shared sents** = unique normalized sentences of A also in B (and as % of A's unique sentences); "
         "**pairs touching B** = A pairs with ≥1 sentence in B; **exact pairs** = A pairs whose unordered sentence pair is also a pair in B.", ""]

    def table(title, subset):
        L.extend([f"## {title}", "", "| A | B | A uniq sents | shared sents | % | A pairs | pairs touching B | % | exact pairs | % |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
        for r in subset:
            f = lambda k, pct=False: ("–" if k not in r else (f"{100 * r[k]:.1f}" if pct else f"{r[k]:,}"))
            L.append(f"| {r['a']} | {r['b']} | {r['a_unique_sents']:,} | {r['shared_sents']:,} | {100 * r['shared_sents_frac_of_a']:.1f} | "
                     f"{f('a_pairs')} | {f('a_pairs_touching_b')} | {f('a_pairs_touching_b_frac', True)} | "
                     f"{f('a_pairs_exact_match_in_b')} | {f('a_pairs_exact_match_frac', True)} |")
        L.append("")

    table("Ranking set (R) and auxiliary set (AUX) vs eval sets (E)",
          [r for r in rows if r["a"] in R_KEYS + AUX_KEYS and r["b"] in E_KEYS])
    table("Dev sets vs E (dev is used only for l* and development)", [r for r in rows if r["a"] in DEV_KEYS])
    table("NLI PCA corpus vs E (label-free; the PCA is fitted on it)", [r for r in rows if r["a"] == NLI_KEY])
    table("Between eval sets (E vs E; never filtered)", [r for r in rows if r["a"] in E_KEYS])
    table("STS-B train vs SICK-R train", [r for r in rows if r["a"] in R_KEYS + AUX_KEYS and r["b"] in R_KEYS + AUX_KEYS])

    L.extend(["## R filter (train_ranked): drop R pairs with either sentence in any E task", "",
              "| set | role | pairs | kept | removed | removed % |", "|---|---|---|---|---|---|"])
    for k, v in r_summary.items():
        L.append(f"| {k} | R | {v['pairs']:,} | {v.get('kept', v['pairs'] - v['removed']):,} | {v['removed']:,} | {100 * v['removed_frac']:.1f} |")
    for k, v in aux_summary.items():
        L.append(f"| {k} | AUX: not in R, shown for the record | {v['pairs']:,} | {v['kept']:,} | {v['removed']:,} | {100 * v['removed_frac']:.1f} |")
    L.extend(["", "## held_out fold filters: pairs of other E tasks kept after removing any pair sharing a sentence with held-out t", "",
              "| held-out t | E∖t pairs kept (of total) | + R kept | = ranking pairs |", "|---|---|---|---|"])
    for t in E_KEYS:
        tot = sum(p2_summary[f"{t}<-{u}"]["pairs"] for u in E_KEYS if u != t)
        fp = p2_summary[f"_fold_{t}_ranking_pairs"]
        L.append(f"| {t} | {fp['E_minus_t_kept']:,} / {tot:,} | {fp['R_kept']:,} | {fp['E_minus_t_kept'] + fp['R_kept']:,} |")
    L.extend(["", f"R removal: {100 * r_summary['_total']['removed_frac']:.1f}% "
              f"(STOP threshold {100 * MAX_R_REMOVAL:.0f}%)."])
    (OUT / "overlap_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-12:]))

    if r_summary["_total"]["removed_frac"] > MAX_R_REMOVAL:
        sys.exit(f"STOP: filter removes {100 * r_summary['_total']['removed_frac']:.1f}% of R (> 50%)")


if __name__ == "__main__":
    main()
