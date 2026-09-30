#!/usr/bin/env python3
"""
End-to-end pipeline for the GNN-ATD dataset, one project at a time:

  1. CLONE     shallow fetch of the exact commit recorded in selected.csv
  2. PRUNE     test / example / demo / sample code removed (prune.py, rules R1-R4)
               and the removal committed locally, so that HEAD = pruned code
               whether a tool reads the working tree or the git history
  3. DESIGNITE labels   (Java: DesigniteJava.jar on the source folder;
               C#: DesigniteConsole.exe on a batch file listing every .csproj
               left after pruning, since it accepts .sln/.csproj/batch, not folders)
  4. ARCAN     graph + features (Docker image, writeDependencyGraph=true)
  5. CHECK     Arcan output must not contain files from pruned directories
  6. DATASET   build_dataset.py -> nodes.csv, edges.csv, meta.json, graph.html
  7. CLEANUP   working copy deleted; raw tool outputs kept in --raw

Every project gets one line in <out>/pipeline_log.jsonl (status, failing
step, durations, prune summary, error tail). Projects already logged as
"ok" are skipped, so the run can be interrupted and resumed.

CONFIGURATION (.env next to this script, or environment variables)
  DESIGNITE_JAVA_JAR   path to DesigniteJava.jar
  DESIGNITE_CS_EXE     path to DesigniteConsole.exe
  DESIGNITE_CS_ARGS    arguments after the exe  (default: analyze -i {src} -o {out},
                       i.e. the same -i/-o interface as DesigniteJava)
  ARCAN_IMAGE          default: ghcr.io/arcan-tech/arcan-2-cli-trial:latest
  ARCAN_LANG_JAVA      default: JAVA
  ARCAN_LANG_CSHARP    default: CSHARP   (check with: arcan analyse -h)

Usage
  python run_pipeline.py --lang java --selected data/java/selected.csv \\
      --n 50 --work D:/gnn_work --raw data/raw/java --out data/dataset/java
"""
import argparse
import csv
import datetime as dt
import json
import os
import random
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prune import prune  # noqa: E402

# raw outputs kept after a successful run (the rest is deleted to save space)
KEEP_DESIGNITE = ("ArchitectureSmells.csv", "TypeMetrics.csv",            # Java
                  "_ArchSmells.csv", "_NamespaceMetrics.csv",               # C#
                  "_ClassMetrics.csv", "AnalysisSummary.csv")
KEEP_ARCAN_SUFFIX = (".graphml", ".csv")


# ------------------------------------------------------------------ utils
def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_env(path):
    if not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(),
                                  v.strip().strip('"').strip("'"))


def rmtree(path):
    """rmtree that also works on Windows read-only files (e.g. .git objects)."""
    def onerror(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if Path(path).exists():
        shutil.rmtree(path, onerror=onerror)


def run(cmd, timeout, name=None):
    """Run a command; return (ok, seconds, output tail)."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        ok, out = p.returncode == 0, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        ok, out = False, f"TIMEOUT after {timeout}s\n{e.stdout or ''}{e.stderr or ''}"
        if name:  # stop the orphan container
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    except FileNotFoundError as e:
        ok, out = False, f"command not found: {e}"
    return ok, round(time.time() - t0, 1), clip(out)


def clip(text, n=1500):
    """Keep head and tail: tools often print the actual error first and a
    long usage/help text afterwards."""
    text = text.strip()
    return text if len(text) <= 2 * n else f"{text[:n]}\n[...]\n{text[-n:]}"


def safe_name(full_name):
    return full_name.replace("/", "__")


# ------------------------------------------------------------------ steps
def clone(url, sha, dest):
    rmtree(dest)
    dest.mkdir(parents=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    for cmd in (["git", "init", "-q"],
                ["git", "remote", "add", "origin", url],
                ["git", "fetch", "-q", "--depth", "1", "origin", sha],
                ["git", "checkout", "-q", "-B", "gnn-atd", "FETCH_HEAD"]):
        p = subprocess.run(cmd, cwd=dest, env=env, timeout=900,
                           capture_output=True, text=True, errors="replace")
        if p.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[:3])}: {p.stderr.strip()[-500:]}")


def commit_pruning(repo):
    """Commit the removal of pruned directories; return the new HEAD sha."""
    git = ["git", "-c", "user.name=gnn-atd", "-c", "user.email=gnn-atd@localhost",
           "-c", "commit.gpgsign=false"]
    for cmd in (["add", "-A"],
                ["commit", "-q", "--allow-empty", "--no-verify",
                 "-m", "GNN-ATD: remove test/example/demo/sample code"]):
        p = subprocess.run(git + cmd, cwd=repo, capture_output=True, text=True,
                           errors="replace", timeout=900)
        if p.returncode != 0:
            raise RuntimeError(f"git {cmd[0]}: {p.stderr.strip()[-500:]}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()


def designite_cs_batch(src, batch_path):
    """Designite (C#) takes a .sln/.csproj or a batch file, not a folder.
    After pruning, the .sln files may reference deleted test projects, so we
    list all remaining .csproj in a batch file ([Projects] section)."""
    projs = sorted(p.resolve() for p in Path(src).rglob("*.csproj"))
    if not projs:
        return 0
    batch_path.write_text("[Projects]\n" + "\n".join(str(p) for p in projs) + "\n",
                          encoding="utf-8")
    return len(projs)


def designite_cmd(lang, src, out):
    if lang == "java":
        return ["java", "-jar", os.environ["DESIGNITE_JAVA_JAR"], "-i", str(src), "-o", str(out)]
    args = os.environ.get("DESIGNITE_CS_ARGS", "analyze -i {src} -o {out}")
    return [os.environ["DESIGNITE_CS_EXE"]] + [
        a.format(src=str(src), out=str(out)) for a in shlex.split(args, posix=False)]


def arcan_cmd(lang, work_proj, name, container):
    lang_flag = os.environ.get("ARCAN_LANG_JAVA", "JAVA") if lang == "java" \
        else os.environ.get("ARCAN_LANG_CSHARP", "CSHARP")
    image = os.environ.get("ARCAN_IMAGE", "ghcr.io/arcan-tech/arcan-2-cli-trial:latest")
    return ["docker", "run", "--rm", "--name", container,
            "-v", f"{(work_proj / 'src').resolve()}:/data/{name}",
            "-v", f"{(work_proj / 'arcan').resolve()}:/out",
            image, "analyse", "-i", f"/data/{name}", "-o", "/out",
            "--all", "-l", lang_flag, "output.writeDependencyGraph=true"]


def find_graphml(arcan_dir, name):
    hits = sorted((arcan_dir / "arcanOutput" / name).glob("dependency-graph-*.graphml")) \
        or sorted(arcan_dir.rglob("dependency-graph-*.graphml"))
    return hits[0] if hits else None


def pruned_leak(arcan_dir, name, removed_dirs):
    """Return pruned paths that nevertheless appear in Arcan's component list."""
    cm = next(iter(sorted(arcan_dir.rglob("component-metrics.csv"))), None)
    if cm is None or not removed_dirs:
        return []
    prefixes = [d["path"].rstrip("/") + "/" for d in removed_dirs]
    leaks = set()
    with open(cm, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            p = (r.get("filePathRelative") or "").replace("\\", "/").lstrip("./")
            for pre in prefixes:
                if p.startswith(pre):
                    leaks.add(pre)
    return sorted(leaks)


def keep_raw(work_proj, raw_proj):
    raw_proj.mkdir(parents=True, exist_ok=True)
    for f in (work_proj / "designite").rglob("*"):
        if f.is_file() and (f.name.endswith(KEEP_DESIGNITE)
                            or f.name.lower().startswith("designitelog")):
            shutil.copy2(f, raw_proj / f"designite_{f.name}")
    for f in (work_proj / "arcan").rglob("*"):
        if f.is_file() and f.name.endswith(KEEP_ARCAN_SUFFIX):
            shutil.copy2(f, raw_proj / f"arcan_{f.name}")


def designite_dir(out):
    """Locate Designite's output folder"""
    for pat in ("ArchitectureSmells.csv", "*AnalysisSummary.csv", "*_NamespaceMetrics.csv"):
        hit = next(iter(sorted(out.rglob(pat))), None)
        if hit:
            return hit.parent
    return None


# ------------------------------------------------------------------- main
def process(row, args):
    name = safe_name(row["full_name"])
    wp = args.work / name
    rec = {"full_name": row["full_name"], "head_sha": row["head_sha"], "status": "ok",
           "failed_step": None, "times": {}, "started": dt.datetime.now().isoformat(timespec="seconds")}

    def fail(step, err):
        rec.update(status="error", failed_step=step, error=clip(err))
        return rec

    try:
        # 1 clone
        t0 = time.time()
        try:
            clone(row["html_url"] + ".git", row["head_sha"], wp / "src")
        except Exception as e:  # noqa: BLE001
            return fail("clone", str(e))
        rec["times"]["clone"] = round(time.time() - t0, 1)

        # 2 prune
        t0 = time.time()
        rep = prune(wp / "src", args.lang, apply=True)
        rec["times"]["prune"] = round(time.time() - t0, 1)
        rec["prune"] = {k: rep[k] for k in ("source_files_before", "source_files_removed",
                                            "removed_share", "removed_by_rule",
                                            "residual_test_files")}
        (wp / "prune_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        if rep["source_files_after"] == 0:
            return fail("prune", "no source files left after pruning")
        try:
            rec["pruned_commit"] = commit_pruning(wp / "src")
        except Exception as e:  # noqa: BLE001
            return fail("prune", f"commit of pruning failed: {e}")

        # 3 Designite
        (wp / "designite").mkdir(exist_ok=True)
        d_input = wp / "src"
        if args.lang == "csharp":
            d_input = wp / "designite_projects.batch"
            n_proj = designite_cs_batch(wp / "src", d_input)
            rec["csproj"] = n_proj
            if n_proj == 0:
                return fail("designite", "no .csproj file after pruning (e.g. Unity project "
                                         "without committed project files)")
        ok, sec, out = run(designite_cmd(args.lang, d_input, wp / "designite"), args.timeout)
        rec["times"]["designite"] = sec
        ddir = designite_dir(wp / "designite")
        if not ok or ddir is None:
            return fail("designite", out)

        # 4 Arcan
        (wp / "arcan").mkdir(exist_ok=True)
        container = f"arcan-{name}".lower()[:60]
        acmd = arcan_cmd(args.lang, wp, name, container)
        rec["arcan_cmd"] = " ".join(acmd)
        ok, sec, out = run(acmd, args.timeout, container)
        rec["times"]["arcan"] = sec
        graph = find_graphml(wp / "arcan", name)
        if not ok or graph is None:
            return fail("arcan", out)

        # 5 check that pruned code did not re-enter the analysis
        leaks = pruned_leak(wp / "arcan", name, rep["removed_dirs"])
        if leaks:
            return fail("check", f"Arcan analysed pruned directories: {leaks[:5]}")

        # 6 dataset
        cmd = [sys.executable, str(HERE / "build_dataset.py"), "--arcan-graph", str(graph),
               "--designite-dir", str(ddir), "--project", name, "--out", str(args.out)]
        if args.no_html:
            cmd.append("--no-html")
        ok, sec, out = run(cmd, 1800)
        rec["times"]["dataset"] = sec
        if not ok:
            return fail("dataset", out)
        meta = json.loads((args.out / name / "meta.json").read_text(encoding="utf-8"))
        rec.update(n_nodes=meta["n_nodes"], n_edges=meta["n_edges"],
                   label_counts=meta["label_counts"], dense_structure=meta["dense_structure"],
                   matched=meta["matching"]["matched"])
        (args.out / name / "prune_report.json").write_text(json.dumps(rep, indent=2),
                                                          encoding="utf-8")
        if not args.no_raw:
            keep_raw(wp, args.raw / name)
        return rec
    finally:
        if not args.keep_work:
            rmtree(wp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["java", "csharp"], required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="dataset folder")
    ap.add_argument("--work", type=Path, required=True, help="temporary working folder")
    ap.add_argument("--raw", type=Path, help="raw tool outputs (default: <out>/../raw_<lang>)")
    ap.add_argument("--n", type=int, help="random sample size (pilot); default: all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", help="process only these full_names (debug)")
    ap.add_argument("--timeout", type=int, default=3600, help="per tool, seconds")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--no-raw", action="store_true", help="do not keep raw tool outputs")
    ap.add_argument("--keep-work", action="store_true", help="keep working copies (debug)")
    ap.add_argument("--retry-errors", action="store_true", help="re-run failed projects")
    ap.add_argument("--env-file", type=Path, default=HERE / ".env")
    args = ap.parse_args()
    load_env(args.env_file)
    args.raw = args.raw or args.out.parent / f"raw_{args.lang}"
    for p in (args.out, args.work, args.raw):
        p.mkdir(parents=True, exist_ok=True)

    with open(args.selected, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.only:
        rows = [r for r in rows if r["full_name"] in set(args.only)]
    elif args.n:
        sample_path = args.out / "sample.csv"
        if sample_path.exists():  # keep the same sample across resumed runs
            with open(sample_path, encoding="utf-8", newline="") as f:
                names = [r["full_name"] for r in csv.DictReader(f)]
            by = {r["full_name"]: r for r in rows}
            rows = [by[n] for n in names if n in by]
        else:
            rows = random.Random(args.seed).sample(sorted(rows, key=lambda r: r["full_name"]),
                                                   min(args.n, len(rows)))
            with open(sample_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    log_path = args.out / "pipeline_log.jsonl"
    done = {}
    if log_path.exists():
        for line in log_path.open(encoding="utf-8"):
            d = json.loads(line)
            done[d["full_name"]] = d
    todo = [r for r in rows if r["full_name"] not in done
            or (args.retry_errors and done[r["full_name"]]["status"] != "ok")]
    log(f"{len(rows)} projects, {len(rows) - len(todo)} already processed, {len(todo)} to go")

    with log_path.open("a", encoding="utf-8") as lf:
        for i, row in enumerate(todo, 1):
            log(f"[{i}/{len(todo)}] {row['full_name']}")
            rec = process(row, args)
            rec["finished"] = dt.datetime.now().isoformat(timespec="seconds")
            lf.write(json.dumps(rec) + "\n")
            lf.flush()
            if rec["status"] == "ok":
                log(f"    ok  nodes={rec['n_nodes']} edges={rec['n_edges']} "
                    f"labels={rec['label_counts']} times={rec['times']}")
            else:
                log(f"    ERROR at {rec['failed_step']}: {rec['error'][:600]}")

    # summary
    final = {}
    for line in log_path.open(encoding="utf-8"):
        d = json.loads(line)
        final[d["full_name"]] = d
    st = {}
    for d in final.values():
        k = d["status"] if d["status"] == "ok" else f"error_{d['failed_step']}"
        st[k] = st.get(k, 0) + 1
    (args.out / "pipeline_summary.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    log(f"done: {st}")


if __name__ == "__main__":
    sys.exit(main())
