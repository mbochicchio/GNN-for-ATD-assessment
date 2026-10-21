#!/usr/bin/env python3
"""
Change-history information for the package graph of one project
(evolutionary coupling), computed from the git history BEFORE the analysed
snapshot only (no information from the future).

INPUT
  --repo      git working copy checked out at the analysed snapshot, with
              history covering the window (e.g. git fetch --shallow-since=...)
  --dataset   project folder produced by graph_builder.py (nodes.csv, edges.csv)

HISTORY CLEANING
  - commits in the window [snapshot date - --months, snapshot date]
  - all commits are kept (no size limit by default: --max-files 0). Merge
    commits are not counted separately: they carry no changes of their own,
    which are already counted in the merged commits (diffing a merge against
    its first parent would count every change twice)
  - renames treated as delete + add (--no-renames)
  - only source files (.java) outside test/example/demo/sample directories
    (same rules as prune.py)
  - optional: --max-files N excludes commits touching more than N source
    files (bulk changes); disabled by default
  - file -> package: package declaration of the file at the snapshot; files
    no longer present are mapped through their directory (Java convention:
    directory = package); files that cannot be mapped are ignored

OUTPUT (in the dataset folder)
  change_features.csv  per graph node:
      chg_freq        share of window commits touching the package
      chg_churn_rel   (lines added + deleted) / LOC of the package
      chg_authors     distinct authors (e-mail) who changed the package
      cc_degree       packages it co-changes with (support >= --min-support)
      cc_nodep_ratio  share of those co-change partners with NO structural
                      dependency in either direction (logical coupling only)
  cochange_edges.csv   undirected co-change pairs with support >= --min-support:
      src, dst, support (commits changing both), conf_max (max of the two
      association-rule confidences), dependency (1 if a structural edge exists
      in either direction)
  cochange_meta.json   window, commits used / excluded, coverage

Usage
  python cochange.py --repo path/to/clone --dataset data/pilot/java/<project>
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prune import rule_for  # noqa: E402

PKG_RE = re.compile(r"^\s*package\s+([A-Za-z_][\w.]*)\s*;", re.M)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def excluded_path(path, lang="java"):
    """True if any directory of the path matches the prune.py exclusion rules."""
    parts = Path(path).parts[:-1]
    return any(rule_for(p, None, [], lang) for p in parts)


def snapshot_packages(repo):
    """file -> package and directory -> package, from the working copy."""
    file_pkg, dir_votes = {}, defaultdict(Counter)
    for rel in git(repo, "ls-files", "*.java").splitlines():
        if excluded_path(rel):
            continue
        try:
            head = (Path(repo) / rel).read_text(encoding="utf-8", errors="ignore")[:65536]
        except OSError:
            continue
        m = PKG_RE.search(head)
        if not m:
            continue
        file_pkg[rel] = m.group(1)
        dir_votes[os.path.dirname(rel)][m.group(1)] += 1
    dir_pkg = {d: c.most_common(1)[0][0] for d, c in dir_votes.items()}
    return file_pkg, dir_pkg


def read_commits(repo, since, until, sha):
    """Yield (author_email, [(added, deleted, path), ...]) for non-merge commits."""
    out = git(repo, "log", sha, "--no-merges", "--no-renames", f"--since={since}",
              f"--until={until}", "--numstat", "--format=@@%H|%ae", "--", "*.java")
    commit = None
    for line in out.splitlines():
        if line.startswith("@@"):
            if commit:
                yield commit
            _, rest = line[2:].split("|", 1) if "|" in line else (line[2:], "")
            commit = (rest.strip().lower(), [])
        elif line.strip() and commit is not None:
            a, d, path = line.split("\t", 2)
            a = int(a) if a.isdigit() else 0
            d = int(d) if d.isdigit() else 0
            commit[1].append((a, d, path))
    if commit:
        yield commit


def compute(repo, dataset, snapshot="HEAD", months=24, max_files=0, min_support=3, verbose=True):
    """Compute change features and co-change edges for one project (see module
    docstring); writes the three output files and returns the meta dict."""
    args = argparse.Namespace(repo=Path(repo), dataset=Path(dataset), snapshot=snapshot,
                              months=months, max_files=max_files, min_support=min_support)

    nodes = pd.read_csv(args.dataset / "nodes.csv")
    edges = pd.read_csv(args.dataset / "edges.csv")
    name_of = dict(zip(nodes["node_id"], nodes["name"]))
    graph_pkgs = set(nodes["name"])
    dep = {frozenset((name_of[s], name_of[t])) for s, t in zip(edges["src"], edges["dst"])}

    sha = git(args.repo, "rev-parse", args.snapshot).strip()
    snap_ts = int(git(args.repo, "log", "-1", "--format=%ct", sha).strip())
    until = dt.datetime.fromtimestamp(snap_ts, dt.timezone.utc)
    since = until - dt.timedelta(days=round(30.44 * args.months))
    oldest = int(git(args.repo, "log", "--reverse", "--format=%ct", sha).splitlines()[0])
    history_ok = oldest <= since.timestamp()

    file_pkg, dir_pkg = snapshot_packages(args.repo)

    def pkg_of(path):
        if excluded_path(path):
            return None
        p = file_pkg.get(path) or dir_pkg.get(os.path.dirname(path))
        return p if p in graph_pkgs else None

    n_used = n_bulk = n_unmapped = 0
    count, churn, authors = Counter(), Counter(), defaultdict(set)
    pair = Counter()
    for author, files in read_commits(args.repo, since.isoformat(), until.isoformat(), sha):
        mapped = [(a, d, pkg_of(p)) for a, d, p in files]
        mapped = [(a, d, p) for a, d, p in mapped if p]
        if not mapped:
            n_unmapped += 1
            continue
        if args.max_files and len(mapped) > args.max_files:
            n_bulk += 1
            continue
        n_used += 1
        pkgs = sorted({p for _, _, p in mapped})
        for a, d, p in mapped:
            churn[p] += a + d
        for p in pkgs:
            count[p] += 1
            authors[p].add(author)
        for x, y in combinations(pkgs, 2):
            pair[(x, y)] += 1

    cc = []
    for (x, y), sup in pair.items():
        if sup >= args.min_support:
            cc.append({"src": x, "dst": y, "support": sup,
                       "conf_max": round(sup / min(count[x], count[y]), 4),
                       "dependency": int(frozenset((x, y)) in dep)})
    cc = pd.DataFrame(cc, columns=["src", "dst", "support", "conf_max", "dependency"])
    partners = defaultdict(set)
    for r in cc.itertuples():
        partners[r.src].add(r.dst)
        partners[r.dst].add(r.src)

    feats = []
    for r in nodes.itertuples():
        p = r.name
        part = partners.get(p, set())
        nodep = sum(frozenset((p, q)) not in dep for q in part)
        feats.append({
            "node_id": r.node_id, "name": p,
            "chg_freq": round(count[p] / n_used, 6) if n_used else 0.0,
            "chg_churn_rel": round(churn[p] / r.LOC, 6) if r.LOC else 0.0,
            "chg_authors": len(authors[p]),
            "cc_degree": len(part),
            "cc_nodep_ratio": round(nodep / len(part), 6) if part else 0.0,
        })
    pd.DataFrame(feats).to_csv(args.dataset / "change_features.csv", index=False)
    # store co-change edges with node ids too (for the GNN)
    nid = {v: k for k, v in name_of.items()}
    cc.insert(0, "dst_id", cc["dst"].map(nid))
    cc.insert(0, "src_id", cc["src"].map(nid))
    cc.to_csv(args.dataset / "cochange_edges.csv", index=False)

    meta = {"snapshot": sha, "window_since": since.isoformat(), "window_until": until.isoformat(),
            "months": args.months, "history_covers_window": history_ok,
            "commits_used": n_used, "commits_bulk_excluded": n_bulk,
            "commits_without_graph_packages": n_unmapped,
            "max_files": args.max_files, "min_support": args.min_support,
            "packages_changed": int(sum(count[p] > 0 for p in graph_pkgs)),
            "packages_in_graph": len(graph_pkgs),
            "cochange_edges": len(cc),
            "cochange_edges_with_dependency": int(cc["dependency"].sum()) if len(cc) else 0}
    (args.dataset / "cochange_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(meta, indent=2))
        if not history_ok:
            print("NOTE: history shorter than the window (young repository or shallow clone)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--snapshot", default="HEAD", help="analysed commit (default HEAD)")
    ap.add_argument("--months", type=int, default=24, help="history window before the snapshot")
    ap.add_argument("--max-files", type=int, default=0,
                    help="exclude commits touching more source files (0 = keep all, default)")
    ap.add_argument("--min-support", type=int, default=3,
                    help="minimum number of commits changing both packages")
    a = ap.parse_args()
    compute(a.repo, a.dataset, a.snapshot, a.months, a.max_files, a.min_support)
    return 0


if __name__ == "__main__":
    sys.exit(main())
