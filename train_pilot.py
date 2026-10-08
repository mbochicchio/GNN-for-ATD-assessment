#!/usr/bin/env python3
"""
Feasibility pilot: can a GNN learn Designite's architectural-smell labels from
Arcan's package dependency graph better than models that use only node-level
features (i.e. no information about the neighbours)?

DATA       one folder per project produced by build_dataset.py
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
  config.json         all parameters, for reproducibility

Usage
  python train_pilot.py --data data/pilot/java --out results/pilot_java
"""

import argparse
import json
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
        "--min-positives",
        type=int,
        default=30,
        help="smell types with fewer positive nodes in the dataset are also "
        "reported separately (macro average without them)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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
        y_te, m_te = te[2].numpy(), te[3].numpy()

        # GraphSAGE and MLP
        for name, graph in (("graphsage", True), ("mlp", False)):
            model = train_nn(graph, tr, va, len(labels), args, seed)
            with torch.no_grad():
                p = torch.sigmoid(model(te[0], te[1])).numpy()
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


if __name__ == "__main__":
    sys.exit(main())
