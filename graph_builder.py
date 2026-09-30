#!/usr/bin/env python3
"""
Build the per-project dataset for the GNN-ATD framework from
  - Arcan       : dependency-graph-*.graphml  (graph structure + node features)
  - Designite   : node labels
                  Java: ArchitectureSmells.csv, TypeMetrics.csv
                  C#  : Designite_<Project>_ArchSmells.csv, _NamespaceMetrics.csv,
                        _ClassMetrics.csv (one set per analysed .csproj, merged)

GRAPH  G = (V, E, F)
  V  Arcan `container` nodes (packages / namespaces), excluding
     - the default package `$`
     - "empty" intermediate containers with no unit of their own
       (e.g. `org`, `org.owasp`: LOC = 0, path '.')
  E  Arcan `containerIsAfferentOf` edges between nodes in V.
     Direction: source depends on target (verified via InferSourceList).
     Attribute: Weight = number of underlying unit-level dependencies.
  F  7 structural features per node (snapshot analysis, no history):
       Ab   AbstractnessMetric
       In   InstabilityMetric
       D    |Ab + In - 1|          (computed, not exported by Arcan)
       FI   FanIn
       FO   FanOut
       LOC  LinesOfCode
       PR   PageRank
     CHO and TACH are NOT used: they are constant (0) on a single snapshot.

LABELS (Designite, 6 node-level smell types)
  CD  Cyclic Dependency       -> Package column + every component listed
                                 as participant in the cycle (Description)
  UD  Unstable Dependency     -> Package column
  AI  Ambiguous Interface     -> Package column
  GC  God Component           -> Package column
  FC  Feature Concentration   -> Package column
  SF  Scattered Functionality -> Package column + every component listed
                                 as realising the same concern (Description)
  Dense Structure is reported by Designite for `<All packages>`, i.e. it is
  a project-level smell: stored as a graph-level flag, not as a node label.

MASK
  label_mask = 1 if the node is also known to Designite (its package appears
  in TypeMetrics.csv / its namespace in *_NamespaceMetrics.csv or
  *_ClassMetrics.csv), 0 otherwise (labels unknown -> excluded from loss).

TARGET FOR RQ2
  atdi = Arcan ComponentAtdIndex (not a feature, reference score only)

OUTPUT (in --out/<project>/)
  nodes.csv   node_id, name, 7 features, 6 labels, label_mask, atdi
  edges.csv   src, dst, weight            (node_id indices)
  meta.json   graph-level info (Dense Structure flag) + matching statistics
  graph.html  interactive view of the graph (requires visualize_graph.py
              in the same directory; disable with --no-html)

Usage
  python graph_builder.py --arcan-graph path/to/dependency-graph-*.graphml \
                          --designite-dir path/to/designite_output \
                          --project DependencyCheck --out data/dataset/java
"""
import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}

FEATURES = ["Ab", "In", "D", "FI", "FO", "LOC", "PR"]
SMELLS = {  # Designite name -> short code (node-level labels)
    "Cyclic Dependency": "CD",
    "Unstable Dependency": "UD",
    "Ambiguous Interface": "AI",
    "God Component": "GC",
    "Feature Concentration": "FC",
    "Scattered Functionality": "SF",
}
LABELS = list(SMELLS.values())
PROJECT_LEVEL = {"Dense Structure"}
ALL_PACKAGES = "<All packages>"
# smells involving several components: the other components are listed
# only in the Description column
PARTICIPANTS_RE = {
    "CD": re.compile(r"components in the cycle are:\s*(.*)", re.S),
    "SF": re.compile(r"components realize the same concern:\s*(.*)", re.S),
}


# ------------------------------------------------------------------ Arcan
def read_graphml(path):
    """Return {node_id: attrs}, [(edge_id, src, dst, attrs)] with attribute
    names (not GraphML key ids)."""
    root = ET.parse(path).getroot()
    key_name = {k.get("id"): k.get("attr.name") for k in root.findall("g:key", NS)}
    graph = root.find("g:graph", NS)
    nodes = {n.get("id"): {key_name[d.get("key")]: d.text for d in n.findall("g:data", NS)}
             for n in graph.findall("g:node", NS)}
    edges = [(e.get("id"), e.get("source"), e.get("target"),
              {key_name[d.get("key")]: d.text for d in e.findall("g:data", NS)})
             for e in graph.findall("g:edge", NS)]
    return nodes, edges


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def arcan_graph(path):
    nodes, edges = read_graphml(path)
    containers = {i: a for i, a in nodes.items() if a.get("labelV") == "container"}

    # units directly belonging to each container
    n_units = {i: 0 for i in containers}
    for _, s, t, a in edges:
        if a.get("labelE") == "belongsTo" and nodes[s].get("labelV") == "unit" and t in n_units:
            n_units[t] += 1

    dropped = {"default_package": [], "empty_container": []}
    keep = []
    for i, a in containers.items():
        if a.get("name") == "$":
            dropped["default_package"].append(a.get("name"))
        elif n_units[i] == 0:
            dropped["empty_container"].append(a.get("name"))
        else:
            keep.append(i)
    keep.sort(key=lambda i: containers[i]["name"])
    idx = {gid: k for k, gid in enumerate(keep)}

    rows = []
    for gid in keep:
        a = containers[gid]
        ab, ins = fnum(a.get("AbstractnessMetric")), fnum(a.get("InstabilityMetric"))
        rows.append({
            "node_id": idx[gid],
            "name": a["name"],
            "Ab": ab,
            "In": ins,
            "D": abs(ab + ins - 1.0),
            "FI": fnum(a.get("FanIn")),
            "FO": fnum(a.get("FanOut")),
            "LOC": fnum(a.get("LinesOfCode")),
            "PR": fnum(a.get("PageRank")),
            "atdi": fnum(a.get("ComponentAtdIndex")),
            "n_units": n_units[gid],
        })

    E = []
    for _, s, t, a in edges:
        if a.get("labelE") == "containerIsAfferentOf" and s in idx and t in idx and s != t:
            E.append((idx[s], idx[t], fnum(a.get("Weight"), 1.0)))
    return rows, sorted(set(E)), dropped


# -------------------------------------------------------------- Designite
def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pkg_col(row):
    # DesigniteJava: "Package"; Designite (C#): "Namespace"
    return (row.get("Package") or row.get("Namespace") or "").strip()


def designite_files(designite_dir):
    """Return (smell rows, set of known packages/namespaces).

    DesigniteJava : ArchitectureSmells.csv + TypeMetrics.csv (one file each)
    Designite C#  : one file set per analysed .csproj, i.e.
                    Designite_<Project>_ArchSmells.csv,
                    Designite_<Project>_NamespaceMetrics.csv,
                    Designite_<Project>_ClassMetrics.csv
                    (no ArchSmells file when a project has no smells)."""
    d = Path(designite_dir)
    if (d / "ArchitectureSmells.csv").exists():
        smells = read_csv(d / "ArchitectureSmells.csv")
        known = {pkg_col(r) for r in read_csv(d / "TypeMetrics.csv")}
    else:
        smells = [r for f in sorted(d.glob("*_ArchSmells.csv")) for r in read_csv(f)]
        known = {pkg_col(r) for pat in ("*_NamespaceMetrics.csv", "*_ClassMetrics.csv")
                 for f in sorted(d.glob(pat)) for r in read_csv(f)}
    return smells, known - {""}


def designite_labels(designite_dir):
    smells, known = designite_files(designite_dir)
    labels, project_level, unknown_smells = {}, set(), set()
    for r in smells:
        smell, pkg = r["Smell"].strip(), pkg_col(r)
        if smell in PROJECT_LEVEL or pkg.startswith("<All"):
            project_level.add(smell)
            continue
        code = SMELLS.get(smell)
        if code is None:
            unknown_smells.add(smell)
            continue
        targets = {pkg}
        if code in PARTICIPANTS_RE:
            m = PARTICIPANTS_RE[code].search(r.get("Description", ""))
            if m:
                # list separator: ";" in DesigniteJava, "'" in Designite (C#)
                targets |= {p.strip().rstrip(".") for p in re.split(r"[;']", m.group(1))
                            if p.strip()}
        for p in targets:
            labels.setdefault(p, set()).add(code)
    return labels, known, project_level, unknown_smells


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arcan-graph", type=Path, required=True)
    ap.add_argument("--designite-dir", type=Path, required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-html", action="store_true", help="do not write graph.html")
    args = ap.parse_args()

    rows, edges, dropped = arcan_graph(args.arcan_graph)
    labels, known, project_level, unknown = designite_labels(args.designite_dir)

    names = {r["name"] for r in rows}
    for r in rows:
        smells = labels.get(r["name"], set())
        for c in LABELS:
            r[c] = int(c in smells)
        r["label_mask"] = int(r["name"] in known)

    out = args.out / args.project
    out.mkdir(parents=True, exist_ok=True)
    cols = ["node_id", "name"] + FEATURES + LABELS + ["label_mask", "atdi", "n_units"]
    with open(out / "nodes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(out / "edges.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "weight"])
        w.writerows(edges)

    labelled_missing = sorted(set(labels) - names)
    meta = {
        "project": args.project,
        "n_nodes": len(rows),
        "n_edges": len(edges),
        "graph_level_smells": sorted(project_level),
        "dense_structure": int("Dense Structure" in project_level),
        "label_counts": {c: sum(r[c] for r in rows) for c in LABELS},
        "matching": {
            "arcan_nodes": len(rows),
            "designite_packages": len(known - {"$"}),
            "matched": len(names & known),
            "arcan_only": sorted(names - known),
            "designite_only": sorted(known - names - {"$"}),
            "smelly_packages_not_in_graph": labelled_missing,
        },
        "dropped_containers": dropped,
        "unknown_designite_smells": sorted(unknown),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_nodes", "n_edges", "label_counts", "dense_structure")}))
    m = meta["matching"]
    print(f"matched {m['matched']}/{m['arcan_nodes']} Arcan nodes; "
          f"Designite-only {len(m['designite_only'])}; "
          f"smelly packages lost {len(m['smelly_packages_not_in_graph'])}")

    if not args.no_html:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from visualize_graph import write_html
        except ImportError:
            print("WARNING: visualize_graph.py not found next to this script; graph.html skipped")
        else:
            print(f"written {write_html(out)}")


if __name__ == "__main__":
    sys.exit(main())
