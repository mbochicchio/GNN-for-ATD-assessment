#!/usr/bin/env python3
"""
Add change-history information (cochange.py) to every project of an existing
dataset, without re-running Arcan or Designite.

For each project folder in --data (output of run_pipeline.py):
  1. fetch the analysed commit (head_sha in --selected) with the history of the
     --months preceding it (git fetch --shallow-since, plus a 30-day margin)
  2. run cochange.compute -> change_features.csv, cochange_edges.csv,
     cochange_meta.json in the project folder
  3. delete the clone
Projects already processed (cochange_meta.json present) are skipped, so the run
can be interrupted and resumed; every outcome is appended to
<data>/history_log.jsonl.

Usage
  python add_history.py --data data/pilot/java --selected data/java/selected.csv \\
      --work data/pilot/history_work
"""
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cochange import compute  # noqa: E402
from run_pipeline import rmtree  # noqa: E402


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def fetch(url, sha, since, dest):
    rmtree(dest)
    dest.mkdir(parents=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    for cmd in (["git", "init", "-q"],
                ["git", "remote", "add", "origin", url],
                ["git", "fetch", "-q", f"--shallow-since={since}", "origin", sha],
                ["git", "checkout", "-q", "FETCH_HEAD"]):
        p = subprocess.run(cmd, cwd=dest, env=env, capture_output=True, text=True,
                           errors="replace", timeout=1800)
        if p.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:3])}: {p.stderr.strip()[-400:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dataset folder (one subfolder per project)")
    ap.add_argument("--selected", type=Path, required=True, help="selected.csv with head_sha, head_commit_date")
    ap.add_argument("--work", type=Path, required=True, help="temporary folder for the clones")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--redo", action="store_true", help="recompute projects already processed")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    with open(args.selected, encoding="utf-8-sig", newline="") as f:
        sel = {r["full_name"].replace("/", "__"): r for r in csv.DictReader(f)}
    projects = sorted(d for d in args.data.iterdir()
                      if d.is_dir() and (d / "nodes.csv").exists())
    todo = [d for d in projects if args.redo or not (d / "cochange_meta.json").exists()]
    log(f"{len(projects)} projects, {len(projects) - len(todo)} already done, {len(todo)} to go")

    log_path = args.data / "history_log.jsonl"
    with log_path.open("a", encoding="utf-8") as lf:
        for i, d in enumerate(todo, 1):
            row = sel.get(d.name)
            rec = {"project": d.name}
            t0 = time.time()
            if row is None:
                rec.update(status="error", error="not in selected.csv")
            else:
                snap = dt.datetime.fromisoformat(row["head_commit_date"].replace("Z", "+00:00"))
                since = (snap - dt.timedelta(days=round(30.44 * args.months) + 30)).date().isoformat()
                clone = args.work / d.name
                try:
                    log(f"[{i}/{len(todo)}] {d.name}")
                    fetch(row["html_url"] + ".git", row["head_sha"], since, clone)
                    meta = compute(clone, d, "HEAD", args.months, args.max_files,
                                   args.min_support, verbose=False)
                    rec.update(status="ok", commits=meta["commits_used"],
                               packages_changed=meta["packages_changed"],
                               packages=meta["packages_in_graph"],
                               cochange_edges=meta["cochange_edges"],
                               logical_only=meta["cochange_edges"] - meta["cochange_edges_with_dependency"])
                    log(f"    ok  commits={meta['commits_used']} changed={meta['packages_changed']}/"
                        f"{meta['packages_in_graph']} cochange_edges={meta['cochange_edges']}")
                except Exception as e:  # noqa: BLE001
                    rec.update(status="error", error=str(e)[-500:])
                    log(f"    ERROR {str(e)[-200:]}")
                finally:
                    rmtree(clone)
            rec["seconds"] = round(time.time() - t0, 1)
            lf.write(json.dumps(rec) + "\n")
            lf.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
