#!/usr/bin/env python3
"""
Remove test, example, demo and sample code from a working copy before running
Designite and Arcan, so that both tools analyse exactly the same files.

EXCLUSION RULES (directory level; matching is case-insensitive)
  R1  Arcan 2 default filter patterns (Arcan CLI manual, `generate filters`):
      a directory is excluded if its name
        - is  test | testing | tests | example | examples | demo | demos
        - ends with  -<kw> | _<kw>        (e.g. integration-test, core_tests)
        - contains   <kw>- | <kw>_        (e.g. test-utils, examples_web)
      with <kw> in {test, testing, tests, example, examples, demo, demos}
      extended with {sample, samples} for consistency with the keyword
      filter used in repository mining (F6).
  R2  Java build-system conventions (Maven / Gradle):
        src/it, testFixtures, integrationTest(s), functionalTest(s)
  R3  C# test-project naming conventions:
        *.Test(s), *.UnitTest(s), *.IntegrationTest(s), *.FunctionalTest(s),
        *.Spec(s), and names ending in UnitTests / IntegrationTests
      (*.Testing is NOT excluded: it usually denotes a shipped test-support
       library, e.g. Polly.Testing, i.e. production code)
  R4  C# project content: a directory containing a .csproj that references
      a test framework (Microsoft.NET.Test.Sdk, xunit, NUnit, MSTest)
  No file-level rule is applied (e.g. *Test.java), to avoid removing
  production classes such as TestUtils in a testing library.

  Benchmarks are kept: they are neither tests nor examples.

Usage (dry run, nothing is deleted):
  python prune.py path/to/repo --lang java
  python prune.py path/to/repo --lang csharp --apply     # actually delete
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

EXT = {"java": ".java", "csharp": ".cs"}

_KW = r"(test|testing|tests|example|examples|demo|demos|sample|samples)"
R1 = re.compile(rf"^{_KW}$|[-_]{_KW}$|{_KW}[-_]", re.I)
R2_JAVA = re.compile(r"^(it|testfixtures|integrationtests?|functionaltests?)$", re.I)
R3_CS = re.compile(r"(\.(unit|integration|functional)?tests?|\.specs?"
                   r"|(unit|integration|functional)tests?)$", re.I)
R4_PKG = re.compile(r'Include\s*=\s*"(Microsoft\.NET\.Test\.Sdk|xunit[^"]*|NUnit[^"]*|'
                    r'MSTest\.[^"]*)"', re.I)
RESIDUAL = {
    "java": re.compile(r"^\s*import\s+(static\s+)?(org\.junit|org\.testng)\b", re.M),
    "csharp": re.compile(r"^\s*using\s+(Xunit|NUnit\.Framework|"
                         r"Microsoft\.VisualStudio\.TestTools)\b", re.M),
}


def csproj_is_test(dirpath, filenames):
    for fn in filenames:
        if fn.lower().endswith(".csproj"):
            try:
                txt = Path(dirpath, fn).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if R4_PKG.search(txt):
                return True
    return False


def rule_for(name, dirpath, filenames, lang):
    if R1.search(name):
        return "R1_arcan_default"
    if lang == "java" and R2_JAVA.search(name):
        return "R2_java_build"
    if lang == "csharp":
        if R3_CS.search(name):
            return "R3_csharp_naming"
        if csproj_is_test(dirpath, filenames):
            return "R4_csharp_test_project"
    return None


def count_sources(root, ext):
    return sum(1 for _, _, fs in os.walk(root) for f in fs if f.endswith(ext))


def prune(root, lang, apply=False):
    """Find (and optionally delete) excluded directories under `root`.
    Returns a JSON-serialisable report."""
    root = Path(root)
    ext = EXT[lang]
    total_before = count_sources(root, ext)
    removed = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        if ".git" in dirnames:
            dirnames.remove(".git")
        keep = []
        for d in dirnames:
            full = Path(dirpath, d)
            try:
                sub_files = os.listdir(full)
            except OSError:
                sub_files = []
            rule = rule_for(d, full, sub_files, lang)
            if rule:
                removed.append({"path": full.relative_to(root).as_posix(), "rule": rule,
                                "source_files": count_sources(full, ext)})
            else:
                keep.append(d)
        dirnames[:] = keep  # do not descend into excluded directories

    if apply:
        for r in removed:
            shutil.rmtree(root / r["path"], ignore_errors=True)

    removed_files = sum(r["source_files"] for r in removed)
    residual = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git" and
                       (apply or (Path(dirpath, d).relative_to(root).as_posix()
                                  not in {r["path"] for r in removed}))]
        for fn in filenames:
            if fn.endswith(ext):
                p = Path(dirpath, fn)
                try:
                    if RESIDUAL[lang].search(p.read_text(encoding="utf-8", errors="ignore")):
                        residual.append(p.relative_to(root).as_posix())
                except OSError:
                    pass

    by_rule = {}
    for r in removed:
        by_rule[r["rule"]] = by_rule.get(r["rule"], 0) + r["source_files"]
    return {
        "lang": lang,
        "applied": apply,
        "source_files_before": total_before,
        "source_files_removed": removed_files,
        "source_files_after": total_before - removed_files,
        "removed_share": round(removed_files / total_before, 4) if total_before else 0.0,
        "removed_by_rule": by_rule,
        "removed_dirs": removed,
        "residual_test_files": len(residual),
        "residual_examples": residual[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--lang", choices=EXT, required=True)
    ap.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    ap.add_argument("--report", type=Path, help="write the JSON report here")
    args = ap.parse_args()
    rep = prune(args.root, args.lang, args.apply)
    txt = json.dumps(rep, indent=2)
    if args.report:
        args.report.write_text(txt, encoding="utf-8")
    summary = {k: rep[k] for k in ("source_files_before", "source_files_removed",
                                   "removed_share", "removed_by_rule", "residual_test_files")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
