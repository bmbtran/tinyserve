"""Append a verified output to RESULTS.md, stamped with git SHA + UTC time.

Either run a command and capture its output:
    uv run python scripts/record_result.py --milestone M1 --run "uv run pytest tests/cpu -q"
or record an output file another command produced (e.g. a Modal GPU run):
    uv run python scripts/record_result.py --milestone M2 --file results/verify/m2.log \
        --command "uv run modal run modal_app.py::gpu_tests --suite m2" --gpu

GPU results must come from a clean working tree (so the SHA identifies the
code), unless --allow-dirty is passed, which is written into the entry.
Paths under results/ and RESULTS.md / COSTLOG.md do not count as dirty.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "RESULTS.md"
IGNORED_PREFIXES = ("results/", "RESULTS.md", "COSTLOG.md")


def git(*args: str) -> str:
    # rstrip only: porcelain lines start with a meaningful space (" M file").
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.rstrip()


def dirty_files() -> list[str]:
    files = []
    for line in git("status", "--porcelain").splitlines():
        path = line[3:].strip().strip('"')
        if not path.startswith(IGNORED_PREFIXES):
            files.append(path)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", required=True)
    ap.add_argument("--file", help="existing output file to record")
    ap.add_argument("--command", help="the command that produced --file")
    ap.add_argument("--run", help="command to run now; its stdout+stderr is recorded")
    ap.add_argument("--gpu", action="store_true", help="GPU result: require a clean tree")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if bool(a.file) == bool(a.run):
        ap.error("pass exactly one of --file / --run")

    dirty = dirty_files()
    if dirty and a.gpu and not a.allow_dirty:
        print("refusing: working tree is dirty for a GPU result:\n  " + "\n  ".join(dirty))
        return 1

    if a.run:
        proc = subprocess.run(a.run, shell=True, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        output = (proc.stdout + proc.stderr).rstrip()
        command, rc = a.run, proc.returncode
    else:
        output = Path(a.file).read_text(encoding="utf-8", errors="replace").rstrip()
        command, rc = a.command or f"(output file {a.file})", None

    sha = git("rev-parse", "--short", "HEAD").strip() or "no-commit"
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"\n## {a.milestone} — {now} — git `{sha}`" + (" (dirty tree)" if dirty else ""),
        "",
        f"Command: `{command}`" + (f"  (exit code {rc})" if rc is not None else ""),
    ]
    if a.file:
        lines.append(f"Source file: `{a.file}`")
    if dirty:
        lines.append("Uncommitted files at record time: " + ", ".join(f"`{d}`" for d in dirty)
                     + (" (--allow-dirty)" if a.allow_dirty else ""))
    if a.note:
        lines.append(f"Note: {a.note}")
    lines += ["", "```text", output, "```", ""]
    if not RESULTS.exists():
        RESULTS.write_text("# RESULTS\n\nRaw, verified outputs appended by `scripts/record_result.py`. "
                           "Do not edit entries by hand.\n", encoding="utf-8")
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"recorded {a.milestone} at {sha} ({len(output.splitlines())} lines)")
    return 0 if rc in (None, 0) else rc


if __name__ == "__main__":
    sys.exit(main())
