#!/usr/bin/env python3
"""
Feasibility pilot: can a GNN learn Designite's architectural-smell labels from
Arcan's package dependency graph better than models that use only node-level
features (i.e. no information about the neighbours)?

DATA       one folder per project produced by graph_builder.py
           (nodes.csv, edges.csv); nodes with label_mask = 0 are excluded
           from loss and evaluation. Smell types without any positive node
           in the whole dataset are dropped (and reported).
FEATURES   Ab, In, D, FI, FO, LOC, PR; log1p on FI, FO, LOC; standardisation
           fitted on the training projects only.
GRAPH      directed dependency edges: u -> v means "package u depends on v".
SPLIT      project level, 60/20/20 (train/val/test), repeated on --splits
           random splits (seeds --seed, --seed+1, ...) to estimate variance.

MODELS
  graphsage  2 directed GraphSAGE layers: out- and in-neighbours aggregated
             separately with the mean aggregator,
             h' = ReLU(W [h || mean_{u in N_out(v)} h_u || mean_{u in N_in(v)} h_u])
             (N_out: packages v depends on; N_in: packages depending on v;
             full neighbourhood, no sampling: graphs are small)
             + one linear head per smell type (sigmoid output)
  mlp        same architecture without neighbourhood aggregation: each node is
             classified from its own 7 features only (no edges as input)
  rf         Random Forest per smell type (class_weight=balanced)
  GraphSAGE and MLP: weighted binary cross-entropy (pos_weight = neg/pos per
  smell on the training nodes), Adam, early stopping on validation macro PR-AUC.

METRICS    per smell type on the pooled test nodes (label_mask = 1): precision,
           recall, F1 (threshold 0.5) and PR-AUC (average precision). In a split
           where a smell type has no positive test node, its metrics are undefined
           (NaN) and excluded. Macro average: unweighted mean over smell types per
           split, then mean ± std over splits; also reported without smell types
           having fewer than --min-positives positive nodes in the dataset.

OUTPUT (in --out)
  results.csv         split x model x smell metrics
  summary.csv         mean and std over splits
  dataset_stats.json  projects, nodes, edges, prevalence per smell
  dataset_manifest.csv  one row per project with SHA-256 of nodes/edges files
  pr_curves.png       precision-recall curves per smell type (test nodes
                      pooled over splits) with the random baseline
  metrics_per_smell.png  F1 and PR-AUC per smell type, mean +- std over splits
  results_per_project.csv  F1 and PR-AUC of every test project, split and model
  paired.csv          paired comparison GraphSAGE vs MLP / RF per smell type:
                      split-level mean difference and splits won; Wilcoxon
                      signed-rank test on test projects (metric averaged over
                      the splits in which a project was tested), Holm-corrected
                      over the non-rare smell types; paired effect size:
                      matched-pairs rank-biserial correlation and share of
                      projects won / tied / lost
  config.json         all parameters, for reproducibility

MLFLOW (optional, --mlflow)
  One parent run per execution (all parameters, dataset statistics, output
  files and figures, the dataset used with its hash, and - automatically, if
  the script is run inside a git repository - the code commit) with one
  nested run per model (metrics per smell type and macro
  averages, mean and std over splits; per-split values logged as steps).
  Default store: local SQLite file mlflow.db + artifacts in mlartifacts/.
  View with:  mlflow ui --backend-store-uri sqlite:///mlflow.db

Usage
  python train_pilot.py --data data/pilot/java --out results/pilot_java
  python train_pilot.py --data data/pilot/java --out results/pilot_java_150 \
      --mlflow --run-name directed-150
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

FEATURES = ["Ab", "In", "D", "FI", "FO", "LOC", "PR"]
LOG_FEATURES = ["FI", "FO", "LOC"]
ALL_LABELS = ["CD", "UD", "AI", "GC", "FC", "SF"]


# ------------------------------------------------------------------- data
def load_projects(root):
    projects = []
    for d in sorted(Path(root).iterdir()):
        if not (d / "nodes.csv").exists() or not (d / "edges.csv").exists():
            continue
        nodes = (
            pd.read_csv(d / "nodes.csv").sort_values("node_id").reset_index(drop=True)
        )
        if len(nodes) < 2:
            continue
        edges = pd.read_csv(d / "edges.csv")
        projects.append({"name": d.name, "nodes": nodes, "edges": edges})
    return projects


def directed_edge_index(edges, n):
    """Edges as (src, dst) = (dependant, dependee); duplicates and self-loops removed."""
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    e = np.unique(edges[["src", "dst"]].to_numpy(), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    assert e.max() < n
    return e.T.astype(np.int64)


def raw_features(nodes):
    x = nodes[FEATURES].to_numpy(dtype=np.float64).copy()
    for f in LOG_FEATURES:
        i = FEATURES.index(f)
        x[:, i] = np.log1p(np.clip(x[:, i], 0, None))
    return x


def merge(projects, labels, mean, std):
    """Concatenate several graphs into one disjoint graph (full-batch)."""
    xs, ys, ms, es, off = [], [], [], [], 0
    for p in projects:
        n = len(p["nodes"])
        xs.append((raw_features(p["nodes"]) - mean) / std)
        ys.append(p["nodes"][labels].to_numpy(dtype=np.float32))
        ms.append(p["nodes"]["label_mask"].to_numpy(dtype=bool))
        es.append(directed_edge_index(p["edges"], n) + off)
        off += n
    return (
        torch.tensor(np.concatenate(xs), dtype=torch.float32),
        torch.tensor(np.concatenate(es, 1), dtype=torch.long),
        torch.tensor(np.concatenate(ys)),
        torch.tensor(np.concatenate(ms)),
    )


# ----------------------------------------------------------------- models
def mean_aggregate(x, index_from, index_to):
    """Mean of x[index_from] grouped by index_to (zero for nodes without neighbours)."""
    agg = torch.zeros_like(x).index_add_(0, index_to, x[index_from])
    deg = (
        torch.zeros(x.size(0), device=x.device)
        .index_add_(0, index_to, torch.ones(index_to.numel(), device=x.device))
        .clamp(min=1)
    )
    return agg / deg.unsqueeze(1)


class DirectedSAGELayer(nn.Module):
    """Directed GraphSAGE layer (mean aggregator), edges u -> v = u depends on v:
    h' = W [h || mean over out-neighbours || mean over in-neighbours]."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(3 * d_in, d_out)

    def forward(self, x, edge_index):
        src, dst = edge_index
        out_mean = mean_aggregate(x, dst, src)  # for each v: packages v depends on
        in_mean = mean_aggregate(x, src, dst)  # for each v: packages depending on v
        return self.lin(torch.cat([x, out_mean, in_mean], 1))


class Net(nn.Module):
    """2-layer encoder + one linear head per smell type (as one K-output layer:
    each output unit has its own weights, i.e. independent heads)."""

    def __init__(self, d_in, hidden, n_labels, dropout, graph=True):
        super().__init__()
        self.graph = graph
        if graph:
            self.l1, self.l2 = DirectedSAGELayer(d_in, hidden), DirectedSAGELayer(
                hidden, hidden
            )
        else:
            self.l1, self.l2 = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden)
        self.heads = nn.Linear(hidden, n_labels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if self.graph:
            h = self.drop(torch.relu(self.l1(x, edge_index)))
            h = self.drop(torch.relu(self.l2(h, edge_index)))
        else:
            h = self.drop(torch.relu(self.l1(x)))
            h = self.drop(torch.relu(self.l2(h)))
        return self.heads(h)  # logits


def macro_prauc(y, p, m):
    vals = []
    for k in range(y.shape[1]):
        yk = y[m, k]
        if yk.sum() > 0:
            vals.append(average_precision_score(yk, p[m, k]))
    return float(np.mean(vals)) if vals else 0.0


def train_nn(graph, tr, va, n_labels, args, seed):
    torch.manual_seed(seed)
    x, ei, y, m = tr
    model = Net(x.size(1), args.hidden, n_labels, args.dropout, graph=graph)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ym = y[m]
    pos = ym.sum(0)
    pos_weight = ((ym.size(0) - pos) / pos.clamp(min=1)).clamp(max=args.max_pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best, best_state, wait = -1.0, None, 0
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x, ei)[m], ym).mean()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(va[0], va[1])).numpy()
        score = macro_prauc(va[2].numpy(), pv, va[3].numpy())
        if score > best + 1e-6:
            best, wait = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate(y, p, m, labels):
    rows = []
    for k, lab in enumerate(labels):
        yk, pk = y[m, k], p[m, k]
        n_pos = int(yk.sum())
        if n_pos == 0:
            # no positive test node: metrics undefined (NaN), excluded from all averages
            pr = rc = f1 = ap = np.nan
        else:
            pr, rc, f1, _ = precision_recall_fscore_support(
                yk, (pk >= 0.5).astype(int), average="binary", zero_division=0
            )
            ap = average_precision_score(yk, pk)
        rows.append(
            {
                "smell": lab,
                "n_test": int(m.sum()),
                "n_pos_test": n_pos,
                "precision": pr,
                "recall": rc,
                "f1": f1,
                "pr_auc": ap,
            }
        )
    return rows


# ------------------------------------------------------ paired comparison
def per_project_metrics(split, model, y, p, m, proj, labels):
    """F1 (threshold 0.5) and average precision of each test project, per smell
    type; projects without positive (labelled) nodes for a smell are skipped."""
    rows = []
    for name in pd.unique(proj):
        sel = (proj == name) & m
        for k, lab in enumerate(labels):
            yk, pk = y[sel, k], p[sel, k]
            if yk.sum() == 0:
                continue
            _, _, f1, _ = precision_recall_fscore_support(
                yk, (pk >= 0.5).astype(int), average="binary", zero_division=0
            )
            rows.append(
                {
                    "split": split,
                    "project": name,
                    "model": model,
                    "smell": lab,
                    "n_pos": int(yk.sum()),
                    "f1": f1,
                    "pr_auc": average_precision_score(yk, pk),
                }
            )
    return rows


MIN_PAIRS = 6  # minimum number of non-tied project pairs for test and effect size


def paired_effect(x, y):
    """Effect size for paired samples (pairs with zero difference are dropped,
    as in the Wilcoxon test):
    - rank-biserial correlation r = (T+ - T-) / (T+ + T-), with T+/T- the sums of
      the ranks of |x - y| for positive/negative differences; r in [-1, 1],
      r > 0 means x better;
    - share of pairs won, tied and lost by x.
    r is not computed (NaN) with fewer than MIN_PAIRS non-tied pairs."""
    from scipy.stats import rankdata

    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    won, tied, lost = (d > 0).mean(), (d == 0).mean(), (d < 0).mean()
    nz = d[d != 0]
    if len(nz) < MIN_PAIRS:  # too few non-tied pairs: effect size not meaningful
        return np.nan, won, tied, lost
    ranks = rankdata(np.abs(nz))
    t_pos, t_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return float((t_pos - t_neg) / (t_pos + t_neg)), won, tied, lost


def effect_magnitude(r):
    a = abs(r)  # conventional thresholds for correlation-type effect sizes
    return (
        "negligible"
        if a < 0.1
        else "small" if a < 0.3 else "medium" if a < 0.5 else "large"
    )


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    q = p[ok][idx] * (ok.sum() - np.arange(ok.sum()))
    q = np.minimum(np.maximum.accumulate(q), 1.0)
    tmp = np.empty_like(q)
    tmp[idx] = q
    out[ok] = tmp
    return out


def paired_comparison(df, proj_df, labels, order, rare):
    """GraphSAGE vs each baseline, per smell type and metric (F1, PR-AUC):
    - mean_diff, splits_won: split-level difference (same test projects);
    - Wilcoxon signed-rank test and paired effect size (rank-biserial correlation,
      share of projects won/tied/lost) on project-level pairs: each test project
      contributes one pair, its metric averaged over the splits in which it was in
      the test set (so that pairs are independent); Holm correction over all
      tests of the non-rare smell types."""
    from scipy.stats import wilcoxon

    rows = []
    if "graphsage" not in order:
        return pd.DataFrame(rows)
    for base in [m for m in order if m != "graphsage"]:
        for c in ("f1", "pr_auc"):
            for lab in labels:
                g = df[(df["smell"] == lab) & (df["model"] == "graphsage")].set_index(
                    "split"
                )[c]
                b = df[(df["smell"] == lab) & (df["model"] == base)].set_index("split")[
                    c
                ]
                d = (g - b).dropna()
                pp = proj_df[proj_df["smell"] == lab]
                pg = pp[pp["model"] == "graphsage"].groupby("project")[c].mean()
                pb = pp[pp["model"] == base].groupby("project")[c].mean()
                common = pg.index.intersection(pb.index)
                xg, xb = pg[common].to_numpy(), pb[common].to_numpy()
                pval = np.nan
                if int((xg != xb).sum()) >= MIN_PAIRS:
                    pval = float(wilcoxon(xg, xb).pvalue)
                r_rb, won, tied, lost = paired_effect(xg, xb)
                rows.append(
                    {
                        "baseline": base,
                        "metric": c,
                        "smell": lab,
                        "rare": lab in rare,
                        "mean_diff": float(d.mean()) if len(d) else np.nan,
                        "splits_won": int((d > 0).sum()),
                        "splits": int(len(d)),
                        "n_projects": int(len(common)),
                        "project_mean_diff": (
                            float((xg - xb).mean()) if len(common) else np.nan
                        ),
                        "p_value": pval,
                        "r_rb": r_rb,
                        "projects_won": won,
                        "projects_tied": tied,
                        "projects_lost": lost,
                        "effect": effect_magnitude(r_rb) if _finite(r_rb) else "",
                    }
                )
    out = pd.DataFrame(rows)
    main = ~out["rare"]
    out["p_holm"] = np.nan
    out.loc[main, "p_holm"] = holm(out.loc[main, "p_value"])
    return out


def print_paired(paired):
    if paired.empty:
        return
    print(
        "\nPaired comparison: GraphSAGE vs baseline (Wilcoxon signed-rank on test projects, "
        "Holm-corrected; paired effect size)"
    )
    print(
        f"{'baseline':<9}{'metric':<8}{'smell':<6}{'diff':>8}{'splits':>8}{'proj':>6}"
        f"{'p':>10}{'p_holm':>10}{'r_rb':>7}{'won/tie/lost':>15}  effect"
    )
    for r in paired.itertuples():
        p = f"{r.p_value:.2g}" if _finite(r.p_value) else "n/a"
        ph = f"{r.p_holm:.2g}" if _finite(r.p_holm) else ("rare" if r.rare else "n/a")
        rr = f"{r.r_rb:+.2f}" if _finite(r.r_rb) else "n/a"
        eff = r.effect if r.effect else "(too few non-tied pairs)"
        wtl = (
            f"{r.projects_won:.0%}/{r.projects_tied:.0%}/{r.projects_lost:.0%}"
            if _finite(r.projects_won)
            else "n/a"
        )
        diff = f"{r.mean_diff:+.3f}" if _finite(r.mean_diff) else "n/a"
        print(
            f"{r.baseline:<9}{r.metric:<8}{r.smell:<6}{diff:>8}{r.splits_won:>5}/{r.splits:<2}"
            f"{r.n_projects:>6}{p:>10}{ph:>10}{rr:>7}{wtl:>15}  {eff}"
        )
    print(
        "(diff/splits: split-level mean difference and splits won; proj: test projects compared;\n"
        " r_rb: matched-pairs rank-biserial correlation, > 0 means GraphSAGE better;\n"
        " won/tie/lost: share of test projects where GraphSAGE is better/equal/worse)"
    )


# ------------------------------------------------- dataset manifest / figures
def _sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(projects, root):
    """One row per project: size and SHA-256 of nodes.csv / edges.csv."""
    rows = []
    for p in projects:
        d = Path(root) / p["name"]
        rows.append(
            {
                "project": p["name"],
                "n_nodes": len(p["nodes"]),
                "n_edges": len(p["edges"]),
                "nodes_sha256": _sha256(d / "nodes.csv"),
                "edges_sha256": _sha256(d / "edges.csv"),
            }
        )
    return pd.DataFrame(rows).sort_values("project").reset_index(drop=True)


def dataset_hash(manifest):
    import hashlib

    payload = "".join(
        f"{r.project}{r.nodes_sha256}{r.edges_sha256}" for r in manifest.itertuples()
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


MODEL_STYLE = {
    "graphsage": ("GraphSAGE", "#1f77b4"),
    "mlp": ("MLP", "#ff7f0e"),
    "rf": ("Random Forest", "#2ca02c"),
}


def make_figures(preds, df, labels, order, stats, out):
    """pr_curves.png: precision-recall curves per smell type (test nodes pooled
    over splits), with the random baseline (prevalence);
    metrics_per_smell.png: F1 and PR-AUC per smell type, mean +- std over splits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    files = []
    k = len(labels)
    fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 3.6), squeeze=False)
    for j, lab in enumerate(labels):
        ax = axes[0, j]
        for mdl in order:
            ys, ps = [], []
            for _s, name, y, p, m in preds:
                if name == mdl:
                    ys.append(y[m, j])
                    ps.append(p[m, j])
            y, p = np.concatenate(ys), np.concatenate(ps)
            if y.sum() == 0:
                continue
            pr, rc, _ = precision_recall_curve(y, p)
            label, color = MODEL_STYLE.get(mdl, (mdl, None))
            ax.plot(
                rc,
                pr,
                color=color,
                lw=1.6,
                label=f"{label} (AP {average_precision_score(y, p):.2f})",
            )
        prev = stats["prevalence"][lab]
        ax.axhline(prev, color="grey", ls="--", lw=1, label=f"random ({prev:.2f})")
        ax.set_title(lab)
        ax.set_xlabel("recall")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 1)
        if j == 0:
            ax.set_ylabel("precision")
        ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
    fig.tight_layout()
    files.append(out / "pr_curves.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(k)
    w = 0.8 / max(len(order), 1)
    for ax, c, title in (
        (axes[0], "f1", "F1 (threshold 0.5)"),
        (axes[1], "pr_auc", "PR-AUC"),
    ):
        g = df.groupby(["smell", "model"])[c].agg(["mean", "std"])
        for i, mdl in enumerate(order):
            means = [
                g.loc[(lab, mdl), "mean"] if (lab, mdl) in g.index else np.nan
                for lab in labels
            ]
            stds = [
                g.loc[(lab, mdl), "std"] if (lab, mdl) in g.index else np.nan
                for lab in labels
            ]
            label, color = MODEL_STYLE.get(mdl, (mdl, None))
            ax.bar(
                x + (i - (len(order) - 1) / 2) * w,
                means,
                w,
                yerr=stds,
                capsize=2,
                color=color,
                label=label,
            )
        if c == "pr_auc":
            ax.scatter(
                x,
                [stats["prevalence"][lab] for lab in labels],
                marker="_",
                s=400,
                color="black",
                zorder=3,
                label="random (prevalence)",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(title)
    handles, labs = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labs, loc="lower center", ncol=len(labs), fontsize=8, frameon=False
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    files.append(out / "metrics_per_smell.png")
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)
    return files


# ----------------------------------------------------------------- MLflow
METRICS = ["precision", "recall", "f1", "pr_auc"]


def _finite(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def log_to_mlflow(args, stats, df, labels, rare, order, manifest, figures, paired):
    """One parent run per execution (parameters, dataset statistics, output
    files) and one nested run per model (metrics per smell type and macro
    averages: mean and std over splits, plus per-split values as steps)."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        sys.exit("--mlflow requires MLflow: pip install mlflow")

    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:  # local artifact store next to the database
        Path(args.artifact_root).mkdir(parents=True, exist_ok=True)
        exp_id = mlflow.create_experiment(
            args.experiment,
            artifact_location=Path(args.artifact_root).resolve().as_uri(),
        )
    else:
        exp_id = exp.experiment_id

    params = {
        k: str(v)
        for k, v in vars(args).items()
        if k
        not in ("mlflow", "tracking_uri", "artifact_root", "experiment", "run_name")
    }
    params.update(
        graph="directed",
        labels_used=",".join(labels),
        labels_dropped=",".join(stats["labels_dropped_no_positives"]) or "-",
        rare_labels=",".join(rare) or "-",
    )
    with mlflow.start_run(experiment_id=exp_id, run_name=args.run_name) as parent:
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "n_projects": stats["projects"],
                "n_nodes": stats["nodes"],
                "n_labelled_nodes": stats["labelled_nodes"],
                "n_edges": stats["edges"],
                "n_smelly_nodes": stats["smelly_nodes"],
            }
        )
        mlflow.log_metrics(
            {f"count_{lab}": stats["smell_counts"][lab] for lab in ALL_LABELS}
        )
        mlflow.log_metrics(
            {f"prevalence_{lab}": stats["prevalence"][lab] for lab in ALL_LABELS}
        )
        pm = {}
        for r in paired.itertuples():
            key = f"paired_{r.baseline}_{r.metric}_{r.smell}"
            for field in (
                "mean_diff",
                "splits_won",
                "p_value",
                "p_holm",
                "r_rb",
                "projects_won",
                "projects_tied",
                "projects_lost",
            ):
                v = getattr(r, field)
                if _finite(v):
                    pm[f"{key}_{field}"] = float(v)
        if pm:
            mlflow.log_metrics(pm)
        # dataset used by this run: per-project manifest with file hashes;
        # its digest identifies the exact data (MLflow "Datasets used")
        import warnings

        with warnings.catch_warnings():  # harmless schema/source-type hints
            warnings.simplefilter("ignore", UserWarning)
            ds = mlflow.data.from_pandas(
                manifest.astype({"n_nodes": float, "n_edges": float}),
                source=Path(args.data).resolve().as_uri(),
                name=f"{Path(args.data).name}-{len(manifest)}p",
            )
            mlflow.log_input(ds, context="training")
        mlflow.log_param("dataset_hash", dataset_hash(manifest))
        for f in (
            "results.csv",
            "summary.csv",
            "dataset_stats.json",
            "config.json",
            "dataset_manifest.csv",
            "results_per_project.csv",
            "paired.csv",
        ):
            if (args.out / f).exists():
                mlflow.log_artifact(str(args.out / f))
        for f in figures:
            mlflow.log_artifact(str(f), artifact_path="figures")

        for mdl in order:
            d = df[df["model"] == mdl]
            with mlflow.start_run(experiment_id=exp_id, run_name=mdl, nested=True):
                mlflow.log_params(
                    {
                        "model": mdl,
                        "graph": "directed" if mdl == "graphsage" else "none",
                    }
                )
                m = {}
                for lab in labels:
                    dl = d[d["smell"] == lab]
                    for c in METRICS:
                        vals = dl[c].dropna()
                        if len(vals):
                            m[f"{c}_{lab}"] = float(vals.mean())
                            m[f"{c}_{lab}_std"] = (
                                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                            )
                        for _, r in dl.iterrows():  # per-split values (step = split)
                            if _finite(r[c]):
                                mlflow.log_metric(
                                    f"split_{c}_{lab}",
                                    float(r[c]),
                                    step=int(r["split"]),
                                )
                for name, frame in (
                    ("macro", d),
                    ("macro_norare", d[~d["smell"].isin(rare)]),
                ):
                    if name == "macro_norare" and not rare:
                        continue
                    per_split = frame.groupby("split")[METRICS].mean()
                    for c in METRICS:
                        vals = per_split[c].dropna()
                        if len(vals):
                            m[f"{name}_{c}"] = float(vals.mean())
                            m[f"{name}_{c}_std"] = (
                                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                            )
                mlflow.log_metrics(m)
        print(
            f"MLflow: run '{parent.info.run_name}' logged in experiment '{args.experiment}' "
            f"({args.tracking_uri})"
        )


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--max-pos-weight", type=float, default=100.0)
    ap.add_argument("--rf-trees", type=int, default=300)
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads for PyTorch (default 1: bit-identical results across runs)",
    )
    ap.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    ap.add_argument(
        "--experiment", default="gnn-atd-pilot", help="MLflow experiment name"
    )
    ap.add_argument(
        "--run-name", default=None, help="MLflow run name (e.g. directed-150)"
    )
    ap.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow backend store (default: local SQLite file)",
    )
    ap.add_argument(
        "--artifact-root",
        default="mlartifacts",
        help="local folder for MLflow artifacts of new experiments",
    )
    ap.add_argument(
        "--min-positives",
        type=int,
        default=30,
        help="smell types with fewer positive nodes in the dataset are also "
        "reported separately (macro average without them)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # reproducibility: deterministic kernels and a fixed number of CPU threads
    # (multi-threaded float reductions change the summation order between runs)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)

    projects = load_projects(args.data)
    if len(projects) < 5:
        sys.exit(
            f"Only {len(projects)} projects found in {args.data}: at least 5 are needed."
        )

    # smell types with at least one positive (masked) node in the dataset
    allnodes = pd.concat([p["nodes"] for p in projects])
    allnodes = allnodes[allnodes["label_mask"] == 1]
    labels = [lab for lab in ALL_LABELS if allnodes[lab].sum() > 0]
    dropped = [lab for lab in ALL_LABELS if lab not in labels]
    stats = {
        "projects": len(projects),
        "nodes": int(sum(len(p["nodes"]) for p in projects)),
        "edges": int(sum(len(p["edges"]) for p in projects)),
        "nodes_per_project": {
            "min": int(min(len(p["nodes"]) for p in projects)),
            "median": float(np.median([len(p["nodes"]) for p in projects])),
            "max": int(max(len(p["nodes"]) for p in projects)),
        },
        "labelled_nodes": int(len(allnodes)),
        "smell_counts": {lab: int(allnodes[lab].sum()) for lab in ALL_LABELS},
        "projects_with_smell": {
            lab: int(
                sum(
                    int(
                        ((p["nodes"]["label_mask"] == 1) & (p["nodes"][lab] == 1)).any()
                    )
                    for p in projects
                )
            )
            for lab in ALL_LABELS
        },
        "smelly_nodes": int((allnodes[ALL_LABELS].sum(axis=1) > 0).sum()),
        "prevalence": {
            lab: round(float(allnodes[lab].mean()), 4) for lab in ALL_LABELS
        },
        "labels_used": labels,
        "labels_dropped_no_positives": dropped,
    }
    (args.out / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    (args.out / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    print(
        f"{len(projects)} projects, {stats['nodes']} nodes ({stats['labelled_nodes']} labelled), "
        f"{stats['edges']} edges"
    )
    print(
        f"smell-affected nodes: {stats['smelly_nodes']} "
        f"({stats['smelly_nodes'] / max(stats['labelled_nodes'], 1):.1%})"
    )
    print(f"{'smell':<6}{'nodes':>8}{'prevalence':>12}{'projects':>10}")
    for lab in ALL_LABELS:
        print(
            f"{lab:<6}{stats['smell_counts'][lab]:>8}{stats['prevalence'][lab]:>12.1%}"
            f"{stats['projects_with_smell'][lab]:>7}/{len(projects)}"
        )
    print(
        f"labels used: {labels}"
        + (f"; dropped (no positives): {dropped}" if dropped else "")
    )

    results = []
    preds = []  # (split, model, y_test, p_test, mask_test) for the figures
    proj_rows = []  # per-project test metrics, for the paired comparison
    for s in range(args.splits):
        seed = args.seed + s
        random.seed(seed)
        np.random.seed(seed)
        idx = list(range(len(projects)))
        random.Random(seed).shuffle(idx)
        n_tr, n_va = round(0.6 * len(idx)), round(0.2 * len(idx))
        tr_p = [projects[i] for i in idx[:n_tr]]
        va_p = [projects[i] for i in idx[n_tr : n_tr + n_va]]
        te_p = [projects[i] for i in idx[n_tr + n_va :]]

        xr = np.concatenate([raw_features(p["nodes"]) for p in tr_p])
        mean, std = xr.mean(0), xr.std(0)
        std[std == 0] = 1.0
        tr, va, te = (merge(ps, labels, mean, std) for ps in (tr_p, va_p, te_p))
        te_proj = np.repeat([q["name"] for q in te_p], [len(q["nodes"]) for q in te_p])
        y_te, m_te = te[2].numpy(), te[3].numpy()

        # GraphSAGE and MLP
        for name, graph in (("graphsage", True), ("mlp", False)):
            model = train_nn(graph, tr, va, len(labels), args, seed)
            with torch.no_grad():
                p = torch.sigmoid(model(te[0], te[1])).numpy()
            preds.append((s, name, y_te, p, m_te))
            proj_rows += per_project_metrics(s, name, y_te, p, m_te, te_proj, labels)
            for r in evaluate(y_te, p, m_te, labels):
                results.append({"split": s, "model": name, **r})

        # Random Forest (one per smell type)
        x_tr, y_tr, m_tr = tr[0].numpy(), tr[2].numpy(), tr[3].numpy()
        p = np.zeros_like(y_te)
        for k in range(len(labels)):
            yk = y_tr[m_tr, k]
            if yk.sum() == 0:
                continue  # no positive training node: predict 0
            rf = RandomForestClassifier(
                n_estimators=args.rf_trees,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(x_tr[m_tr], yk)
            p[:, k] = rf.predict_proba(te[0].numpy())[:, 1]
        preds.append((s, "rf", y_te, p, m_te))
        proj_rows += per_project_metrics(s, "rf", y_te, p, m_te, te_proj, labels)
        for r in evaluate(y_te, p, m_te, labels):
            results.append({"split": s, "model": "rf", **r})

        sp = pd.DataFrame([r for r in results if r["split"] == s])
        print(
            f"split {s}: train {len(tr_p)} / val {len(va_p)} / test {len(te_p)} projects | "
            + " | ".join(
                f"{mdl} F1 {g['f1'].mean():.3f} PR-AUC {g['pr_auc'].mean():.3f}"
                for mdl, g in sp.groupby("model", sort=False)
            )
        )

    df = pd.DataFrame(results)
    df.to_csv(args.out / "results.csv", index=False)
    summ = (
        df.groupby(["model", "smell"])[["precision", "recall", "f1", "pr_auc"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summ.to_csv(args.out / "summary.csv")

    metrics = ["precision", "recall", "f1", "pr_auc"]
    order = [m for m in ("graphsage", "mlp", "rf") if m in set(df["model"])]

    def print_macro(frame, title):
        # per split: unweighted mean over smell types with >= 1 positive test node
        # (NaN rows skipped); then mean and std over splits
        macro = (
            frame.groupby(["model", "split"])[metrics]
            .mean()
            .groupby("model")
            .agg(["mean", "std"])
        )
        print(f"\n{title}")
        print(f"{'model':<10}" + "".join(f"{c:>18}" for c in metrics))
        for mdl in order:
            print(
                f"{mdl:<10}"
                + "".join(
                    f"{macro.loc[mdl, (c, 'mean')]:>11.3f} ± {macro.loc[mdl, (c, 'std')]:.3f}"
                    for c in metrics
                )
            )

    print_macro(
        df,
        "Macro average over smell types (mean ± std over splits; "
        "smell types without positive test nodes excluded per split)",
    )
    rare = [lab for lab in labels if stats["smell_counts"][lab] < args.min_positives]
    if rare and len(rare) < len(labels):
        print_macro(
            df[~df["smell"].isin(rare)],
            f"Macro average without rare smell types {rare} "
            f"(< {args.min_positives} positive nodes in the dataset)",
        )

    for c in ("f1", "pr_auc"):
        tab = df.groupby(["smell", "model"])[c].agg(["mean", "std"])
        print(
            f"\n{c.upper()} per smell type (mean ± std over splits with positive test nodes)"
        )
        print(
            f"{'smell':<6}{'prev.':>7}{'splits':>8}"
            + "".join(f"{m:>17}" for m in order)
        )
        for lab in labels:
            n_spl = int(
                df[(df["smell"] == lab) & (df["model"] == order[0])][c].notna().sum()
            )
            cells = "".join(
                (
                    f"{tab.loc[(lab, m), 'mean']:>10.3f} ± {tab.loc[(lab, m), 'std']:.3f}"
                    if n_spl
                    else f"{'n/a':>17}"
                )
                for m in order
            )
            print(
                f"{lab:<6}{stats['prevalence'][lab]:>7.1%}{n_spl:>5}/{args.splits}{cells}"
            )
    print("(prev. = prevalence = PR-AUC of a random classifier)")
    print(f"\nwritten {args.out / 'results.csv'}, {args.out / 'summary.csv'}")
    proj_df = pd.DataFrame(proj_rows)
    proj_df.to_csv(args.out / "results_per_project.csv", index=False)
    paired = paired_comparison(df, proj_df, labels, order, rare)
    paired.to_csv(args.out / "paired.csv", index=False)
    print_paired(paired)

    manifest = dataset_manifest(projects, args.data)
    manifest.to_csv(args.out / "dataset_manifest.csv", index=False)
    figures = make_figures(preds, df, labels, order, stats, args.out)
    print(f"figures: {', '.join(f.name for f in figures)}")
    if args.mlflow:
        log_to_mlflow(args, stats, df, labels, rare, order, manifest, figures, paired)


if __name__ == "__main__":
    sys.exit(main())
