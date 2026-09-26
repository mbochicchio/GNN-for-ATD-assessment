#!/usr/bin/env python3
"""
Repository mining for the GNN-ATD dataset (Java / C#).

FILTERS (default values; CLI flag in brackets)
  F1  Language: Java or C#                                   [--lang]
  F2  Stars >= 100                                           [--min-stars]
  F3  No forks                                               (search: fork:false)
  F4  No archived repositories                               (search: archived:false)
  F5  Last commit on the default branch within 2 years       [--years]
      (pre-filtered on `pushed` in the search, then checked on
       the actual last commit of the default branch via API)
  F6  Noise keywords in name, topics OR description          [--extra-keywords]
      demo(s), exam(s), example(s), test(s), sample(s)
      matched as whole tokens (camelCase and -_./ split),
      e.g. "latest" or "contest" do NOT match "test"
  F7  Primary-language share >= 80% (GitHub Linguist bytes)  [--min-lang-share]
  F8  >= 10 distinct packages (Java) / namespaces (C#)       [--min-units]
      counted from `package` / `namespace` declarations;
      test, bin, obj, target, build, generated dirs excluded [--no-exclude-tests]
  F9  Manual inspection (NOT automated): fill the
      `manual_check` column of selected.csv

PIPELINE (cheapest stage first)
  S1  GitHub Search API  : F1-F5 (F5 approximated by `pushed`)
  S2  local              : F6
  S3  /languages API     : F7
  S4  /commits API       : F5 (exact, no clone needed if rejected)
      shallow clone      : F8; HEAD SHA recorded to pin the snapshot,
                           clone deleted right after counting
  S3-S4 run in parallel (--workers, default 8) on a random permutation
  of the candidates (--seed, default 42) and stop once --target
  (default 1400, oversampled vs. 1000 to absorb Designite/Arcan failures)
  repositories are accepted.

OUTPUTS (in --out)
  candidates.jsonl  all S1 results (cache; --refresh to redo the search)
  decisions.jsonl   one line per evaluated repo with status and reason
                    (incl. package/namespace names, for duplicate detection);
                    allows interrupting and resuming the run
  selected.csv      accepted repos (SHA, stars, share, #units, description)
  summary.json      run configuration (incl. run date and last-commit
                    cutoff) + counts per status, for the selection funnel

USAGE
  # .env (same directory, never commit it):  GITHUB_TOKEN=ghp_...
  python mine_repos.py --lang java   --out data/java
  python mine_repos.py --lang csharp --out data/csharp
  python mine_repos.py --lang java --out data/java --env-file ~/secrets/gnn.env
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://api.github.com"

LANGS = {
    "java": {
        "query": "java",
        "linguist": "Java",
        "ext": ".java",
        "decl": re.compile(r"^\s*package\s+([A-Za-z_][\w.]*)\s*;", re.M),
    },
    "csharp": {
        "query": "csharp",
        "linguist": "C#",
        "ext": ".cs",
        # covers both block-scoped and file-scoped (C# 10) namespaces
        "decl": re.compile(r"^\s*namespace\s+([A-Za-z_][\w.]*)", re.M),
    },
}

# Keywords from the protocol (+ plural forms)
DEFAULT_KEYWORDS = {
    "demo",
    "demos",
    "exam",
    "exams",
    "example",
    "examples",
    "test",
    "tests",
    "sample",
    "samples",
}

# Directories ignored when counting packages/namespaces
SKIP_DIRS = {
    "test",
    "tests",
    "bin",
    "obj",
    "target",
    "build",
    "generated",
    ".git",
    "node_modules",
}


# --------------------------------------------------------------------- utils
def load_env(path):
    """Minimal .env loader (KEY=VALUE per line, '#' comments, optional quotes).
    Variables already set in the shell take precedence."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def tokens(text):
    """Split on non-alphanumerics and camelCase, lowercase."""
    if not text:
        return set()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", text) if t}


def gh_get(session, url, params=None, is_search=False):
    """GET with primary/secondary rate-limit handling."""
    while True:
        r = session.get(url, params=params, timeout=60)
        if r.status_code in (403, 429):
            if r.headers.get("Retry-After"):
                wait = int(r.headers["Retry-After"])
            elif r.headers.get("X-RateLimit-Remaining") == "0":
                wait = max(int(r.headers["X-RateLimit-Reset"]) - time.time(), 0)
            else:
                wait = 60
            log(f"rate limited, sleeping {wait + 2:.0f}s")
            time.sleep(wait + 2)
            continue
        if r.status_code >= 500:
            time.sleep(10)
            continue
        if is_search:
            time.sleep(2.2)  # search API: 30 req/min authenticated
        return r


# -------------------------------------------------------------- S1: search
def search_slice(session, base_q, start, end, out):
    """Recursively split the creation-date window so that each query
    returns <= 1000 results (hard limit of the Search API)."""
    q = f"{base_q} created:{start.isoformat()}..{end.isoformat()}"
    params = {"q": q, "per_page": 100, "page": 1, "sort": "stars"}
    r = gh_get(session, f"{API}/search/repositories", params, is_search=True)
    r.raise_for_status()
    data = r.json()
    total = data["total_count"]

    if total > 1000 and start < end:
        mid = start + (end - start) / 2
        search_slice(session, base_q, start, mid, out)
        search_slice(session, base_q, mid + dt.timedelta(days=1), end, out)
        return
    if total > 1000:
        log(f"WARNING: {total} results on single day {start}; truncated to 1000")
    if data.get("incomplete_results"):
        log(f"WARNING: incomplete results for {start}..{end}")

    items = data["items"]
    for page in range(2, min(math.ceil(total / 100), 10) + 1):
        params["page"] = page
        r = gh_get(session, f"{API}/search/repositories", params, is_search=True)
        r.raise_for_status()
        items += r.json()["items"]

    for it in items:
        out[it["id"]] = {
            "id": it["id"],
            "full_name": it["full_name"],
            "clone_url": it["clone_url"],
            "html_url": it["html_url"],
            "default_branch": it["default_branch"],
            "description": it.get("description") or "",
            "topics": it.get("topics") or [],
            "stars": it["stargazers_count"],
            "created_at": it["created_at"],
            "pushed_at": it["pushed_at"],
            "fork": it["fork"],
            "archived": it["archived"],
        }
    log(f"{start}..{end}: {total} results (cumulative {len(out)})")


def collect_candidates(session, args, cfg, cutoff):
    base_q = (
        f"language:{cfg['query']} stars:>={args.min_stars} "
        f"fork:false archived:false pushed:>={cutoff.isoformat()}"
    )
    out = {}
    search_slice(session, base_q, args.created_from, args.created_to, out)
    return list(out.values())


# ------------------------------------------------------------ S2: keywords
def keyword_hit(repo, keywords):
    fields = {
        "name": tokens(repo["full_name"].split("/")[1]),
        "topics": (
            set().union(*(tokens(t) for t in repo["topics"]))
            if repo["topics"]
            else set()
        ),
        "description": tokens(repo["description"]),
    }
    for field, toks in fields.items():
        hit = toks & keywords
        if hit:
            return field, sorted(hit)
    return None


# ------------------------------------------------------ S3: language share
def language_share(session, repo, linguist_name):
    r = gh_get(session, f"{API}/repos/{repo['full_name']}/languages")
    if r.status_code != 200:
        return None
    langs = r.json()
    total = sum(langs.values())
    return langs.get(linguist_name, 0) / total if total else 0.0


def last_commit(session, repo):
    """(sha, ISO date) of the last commit on the default branch, via API."""
    r = gh_get(
        session,
        f"{API}/repos/{repo['full_name']}/commits",
        {"sha": repo["default_branch"], "per_page": 1},
    )
    if r.status_code != 200 or not r.json():
        return None
    c = r.json()[0]
    return c["sha"], c["commit"]["committer"]["date"]


# --------------------------------------------- S4: clone, date, packages
def shallow_clone(repo, dest, timeout):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--single-branch",
        "--no-tags",
        "--quiet",
        "--branch",
        repo["default_branch"],
        repo["clone_url"],
        str(dest),
    ]
    subprocess.run(
        cmd,
        check=True,
        timeout=timeout,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out = subprocess.run(
        ["git", "-C", str(dest), "log", "-1", "--format=%H %cI"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return out[0], out[1]


def count_units(root, cfg, exclude_tests):
    """Number of distinct package (Java) / namespace (C#) declarations."""
    names, n_files = set(), 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not (
                exclude_tests
                and (d.lower() in SKIP_DIRS or d.lower().endswith((".tests", ".test")))
            )
            and d != ".git"
        ]
        for fn in filenames:
            if not fn.endswith(cfg["ext"]):
                continue
            n_files += 1
            try:
                with open(
                    os.path.join(dirpath, fn), encoding="utf-8", errors="ignore"
                ) as f:
                    head = f.read(65536)
            except OSError:
                continue
            found = cfg["decl"].findall(head)
            if cfg["ext"] == ".java":
                found = found[:1]
            names.update(found)
    return len(names), n_files, sorted(names)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=LANGS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--target",
        type=int,
        default=1400,
        help="accepted repositories (oversample: tools will fail on some)",
    )
    ap.add_argument("--min-stars", type=int, default=100)
    ap.add_argument("--years", type=float, default=2.0, help="max age of last commit")
    ap.add_argument("--min-lang-share", type=float, default=0.80)
    ap.add_argument("--min-units", type=int, default=10, help="min packages/namespaces")
    ap.add_argument("--extra-keywords", nargs="*", default=[])
    ap.add_argument(
        "--no-exclude-tests",
        action="store_true",
        help="count packages/namespaces in test directories too",
    )
    ap.add_argument(
        "--created-from", type=dt.date.fromisoformat, default=dt.date(2008, 1, 1)
    )
    ap.add_argument("--created-to", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clone-timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8, help="parallel clones")
    ap.add_argument("--refresh", action="store_true", help="re-run the search stage")
    ap.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="file with GITHUB_TOKEN=... (default: ./.env)",
    )
    args = ap.parse_args()
    load_env(args.env_file)

    cfg = LANGS[args.lang]
    args.out.mkdir(parents=True, exist_ok=True)
    cutoff = dt.date.today() - dt.timedelta(days=round(365.25 * args.years))
    keywords = DEFAULT_KEYWORDS | {k.lower() for k in args.extra_keywords}

    def make_session():
        sess = requests.Session()
        sess.headers["Accept"] = "application/vnd.github+json"
        if os.environ.get("GITHUB_TOKEN"):
            sess.headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        return sess

    if not os.environ.get("GITHUB_TOKEN"):
        log("WARNING: no GITHUB_TOKEN, very low rate limits")
    session = make_session()

    # ---- S1
    cand_path = args.out / "candidates.jsonl"
    if cand_path.exists() and not args.refresh:
        candidates = [json.loads(l) for l in cand_path.open(encoding="utf-8")]
        log(f"loaded {len(candidates)} candidates from cache")
    else:
        candidates = collect_candidates(session, args, cfg, cutoff)
        with cand_path.open("w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")
        log(f"S1: {len(candidates)} candidates")

    # ---- resume
    dec_path = args.out / "decisions.jsonl"
    done = {}
    if dec_path.exists():
        for l in dec_path.open(encoding="utf-8"):
            d = json.loads(l)
            done[d["full_name"]] = d
    dec_f = dec_path.open("a", encoding="utf-8")

    def record(repo, status, **extra):
        d = {"full_name": repo["full_name"], "status": status, **extra}
        done[repo["full_name"]] = d
        dec_f.write(json.dumps(d) + "\n")
        dec_f.flush()

    # ---- S2 (all candidates, cheap)
    pool = []
    for repo in candidates:
        if repo["full_name"] in done:
            if done[repo["full_name"]]["status"] != "rejected_keyword":
                pool.append(repo)
            continue
        hit = keyword_hit(repo, keywords)
        if hit:
            record(repo, "rejected_keyword", field=hit[0], keywords=hit[1])
        else:
            pool.append(repo)
    log(f"S2: {len(pool)} candidates after keyword filter")

    # ---- seeded permutation (sort first for determinism)
    pool.sort(key=lambda r: r["full_name"].lower())
    random.Random(args.seed).shuffle(pool)

    tls = threading.local()

    def get_session():
        if not hasattr(tls, "s"):
            tls.s = make_session()
        return tls.s

    def process(repo):
        """S3 -> S4a (API date) -> S4b (shallow clone + count). Returns (status, info)."""
        sess = get_session()
        share = language_share(sess, repo, cfg["linguist"])
        if share is None:
            return "error_languages", {}
        info = {"lang_share": round(share, 4)}
        if share < args.min_lang_share:
            return "rejected_lang_share", info

        lc = last_commit(sess, repo)
        if lc is None:
            return "error_commits", info
        info["head_commit_date"] = lc[1]
        if dt.datetime.fromisoformat(lc[1].replace("Z", "+00:00")).date() < cutoff:
            return "rejected_last_commit", info

        dest = tmp_root / repo["full_name"].replace("/", "__")
        try:
            sha, commit_date = shallow_clone(repo, dest, args.clone_timeout)
            info.update(head_sha=sha, head_commit_date=commit_date)
            n_units, n_files, unit_names = count_units(
                dest, cfg, not args.no_exclude_tests
            )
            # names kept for later near-duplicate detection (Jaccard on name sets)
            info.update(n_units=n_units, n_files=n_files, unit_names=unit_names)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
            return "error_clone", info
        finally:
            shutil.rmtree(dest, ignore_errors=True)
        if n_units < args.min_units:
            return "rejected_units", info
        return "accepted", info

    accepted = sum(1 for d in done.values() if d["status"] == "accepted")
    todo = iter(r for r in pool if r["full_name"] not in done)
    tmp_root = Path(tempfile.mkdtemp(prefix="mine_"))
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            running = {}

            def refill():
                while len(running) < args.workers * 2 and accepted < args.target:
                    repo = next(todo, None)
                    if repo is None:
                        return
                    running[ex.submit(process, repo)] = repo

            refill()
            while running:
                fut = next(as_completed(running))
                repo = running.pop(fut)
                try:
                    status, info = fut.result()
                except Exception as e:  # never lose the whole run for one repo
                    status, info = "error_other", {"error": repr(e)[:200]}
                if status == "accepted" and accepted >= args.target:
                    # in-flight when the target was reached: not recorded, so it
                    # is re-evaluated if the run is resumed with a larger target
                    refill()
                    continue
                record(repo, status, **info)
                if status == "accepted":
                    accepted += 1
                    log(
                        f"[{accepted}/{args.target}] {repo['full_name']} "
                        f"units={info['n_units']} share={info['lang_share']:.2f}"
                    )
                refill()
    finally:
        dec_f.close()
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ---- outputs
    by_name = {c["full_name"]: c for c in candidates}
    with (args.out / "selected.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "full_name",
                "html_url",
                "default_branch",
                "head_sha",
                "head_commit_date",
                "stars",
                "lang_share",
                "n_units",
                "n_files",
                "description",
                "manual_check",
            ]
        )
        for d in done.values():
            if d["status"] != "accepted":
                continue
            c = by_name.get(d["full_name"], {})
            w.writerow(
                [
                    d["full_name"],
                    c.get("html_url"),
                    c.get("default_branch"),
                    d["head_sha"],
                    d["head_commit_date"],
                    c.get("stars"),
                    d["lang_share"],
                    d["n_units"],
                    d["n_files"],
                    c.get("description", ""),
                    "",
                ]
            )

    counts = {}
    for d in done.values():
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    counts["_candidates_S1"] = len(candidates)
    run_config = {
        k: (str(v) if isinstance(v, (Path, dt.date)) else v)
        for k, v in vars(args).items()
        if k != "env_file"
    }
    run_config.update(
        run_date=dt.date.today().isoformat(),
        last_commit_cutoff=cutoff.isoformat(),
        keywords=sorted(keywords),
    )
    (args.out / "summary.json").write_text(
        json.dumps({"config": run_config, "counts": counts}, indent=2), encoding="utf-8"
    )
    log(f"done: {json.dumps(counts)}")
    if accepted < args.target:
        log(f"WARNING: pool exhausted, only {accepted}/{args.target} accepted")


if __name__ == "__main__":
    sys.exit(main())
