#!/usr/bin/env python3
"""
Paired comparison of the same model between two training runs (ablation),
e.g. GraphSAGE with vs without change-history information.

Both runs must use the same projects, splits and seeds (same --data, --seed,
--splits; use --require-history on the baseline run), so that every test
project is evaluated in both runs on the same splits.

For each smell type and metric (F1, PR-AUC), on the test projects (metric
averaged over the splits in which a project was tested):
  mean difference (B - A), Wilcoxon signed-rank test, Holm correction over the
  non-rare smell types, matched-pairs rank-biserial correlation and share of
  projects won / tied / lost by run B.

Usage
  python compare_runs.py --a results/abl_base --b results/abl_change --model graphsage
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pilot import MIN_PAIRS, _finite, effect_magnitude, holm, paired_effect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="output folder of run A (reference)")
    ap.add_argument("--b", type=Path, required=True, help="output folder of run B")
    ap.add_argument("--model", default="graphsage", help="graphsage | mlp | rf")
    ap.add_argument("--min-positives", type=int, default=30,
                    help="smell types with fewer positive nodes are reported as rare")
    ap.add_argument("--out", type=Path, help="CSV output (default: <b>/compare_vs_<a>.csv)")
    args = ap.parse_args()
    from scipy.stats import wilcoxon

    ra = pd.read_csv(args.a / "results_per_project.csv")
    rb = pd.read_csv(args.b / "results_per_project.csv")
    ra, rb = ra[ra["model"] == args.model], rb[rb["model"] == args.model]
    pa = set(zip(ra["split"], ra["project"]))
    pb = set(zip(rb["split"], rb["project"]))
    if pa != pb:
        print(f"WARNING: the runs differ in test projects/splits ({len(pa ^ pb)} differences): "
              "the comparison uses only the common ones; check --seed/--splits/--require-history")

    stats_b = (args.b / "dataset_stats.json")
    rare = []
    if stats_b.exists():
        import json
        counts = json.loads(stats_b.read_text(encoding="utf-8"))["smell_counts"]
        rare = [s for s, c in counts.items() if c < args.min_positives]

    rows = []
    for c in ("f1", "pr_auc"):
        for lab in sorted(set(ra["smell"]) & set(rb["smell"])):
            a = ra[ra["smell"] == lab].groupby("project")[c].mean()
            b = rb[rb["smell"] == lab].groupby("project")[c].mean()
            common = a.index.intersection(b.index)
            xa, xb = a[common].to_numpy(), b[common].to_numpy()
            pval = np.nan
            if int((xa != xb).sum()) >= MIN_PAIRS:
                pval = float(wilcoxon(xb, xa).pvalue)
            r, won, tied, lost = paired_effect(xb, xa)
            rows.append({"metric": c, "smell": lab, "rare": lab in rare, "n_projects": len(common),
                         "mean_a": float(xa.mean()) if len(common) else np.nan,
                         "mean_b": float(xb.mean()) if len(common) else np.nan,
                         "diff": float((xb - xa).mean()) if len(common) else np.nan,
                         "p_value": pval, "r_rb": r, "won": won, "tied": tied, "lost": lost,
                         "effect": effect_magnitude(r) if _finite(r) else ""})
    out = pd.DataFrame(rows)
    main_rows = ~out["rare"]
    out["p_holm"] = np.nan
    out.loc[main_rows, "p_holm"] = holm(out.loc[main_rows, "p_value"])
    dest = args.out or args.b / f"compare_vs_{args.a.name}.csv"
    out.to_csv(dest, index=False)

    print(f"{args.model}: B = {args.b}  vs  A = {args.a}  (test projects, B - A)")
    print(f"{'metric':<8}{'smell':<6}{'proj':>5}{'A':>8}{'B':>8}{'diff':>8}{'p':>10}{'p_holm':>9}"
          f"{'r_rb':>7}{'won/tie/lost':>15}  effect")
    for r in out.itertuples():
        f = lambda v, fmt: format(v, fmt) if _finite(v) else "n/a"  # noqa: E731
        ph = f(r.p_holm, ".2g") if not r.rare else "rare"
        wtl = f"{r.won:.0%}/{r.tied:.0%}/{r.lost:.0%}" if _finite(r.won) else "n/a"
        print(f"{r.metric:<8}{r.smell:<6}{r.n_projects:>5}{f(r.mean_a, '.3f'):>8}{f(r.mean_b, '.3f'):>8}"
              f"{f(r.diff, '+.3f'):>8}{f(r.p_value, '.2g'):>10}{ph:>9}{f(r.r_rb, '+.2f'):>7}{wtl:>15}"
              f"  {r.effect or '(too few non-tied pairs)'}")
    print(f"written {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
