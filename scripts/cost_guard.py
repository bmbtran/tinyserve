"""Refuse to start GPU work once the month's Modal spend reaches the budget.

Spend = max(billing report, sum of COSTLOG.md estimates). The billing API only
reports *complete* intervals and lags a bit, so the local COSTLOG estimate is
a useful lower bound for the current day.

Exit codes: 0 = OK (maybe with a warning), 2 = hard stop (>= $25),
3 = billing unavailable several times in a row (check the dashboard).

    uv run python scripts/cost_guard.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

WARN_USD = 22.0
STOP_USD = 25.0
ROOT = Path(__file__).resolve().parent.parent
COSTLOG = ROOT / "COSTLOG.md"
FAIL_FILE = ROOT / "results" / "tmp" / ".billing_failures"
MAX_FAILS = 3


def _sum_costs(obj) -> float:
    """Sum every value under a key containing 'cost' (shape-agnostic)."""
    total = 0.0
    if isinstance(obj, list):
        for x in obj:
            total += _sum_costs(x)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if "cost" in k.lower() and isinstance(v, (int, float, str)):
                try:
                    total += float(v)
                except ValueError:
                    pass
            elif isinstance(v, (list, dict)):
                total += _sum_costs(v)
    return total


def _billing(args: list[str]) -> float:
    out = subprocess.run(
        [sys.executable, "-m", "modal", "billing", "report", *args, "--json"],
        capture_output=True, text=True, timeout=120, encoding="utf-8",
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-500:])
    return _sum_costs(json.loads(out.stdout or "[]"))


def billing_month_to_date() -> float | None:
    """Completed days this month + completed hours today (may double count a
    little, which errs on the safe side)."""
    try:
        total = _billing(["--for", "this month"]) + _billing(["--for", "today", "-r", "h"])
    except Exception as e:  # noqa: BLE001
        print(f"[cost_guard] WARNING: billing report failed: {e}")
        n = int(FAIL_FILE.read_text()) + 1 if FAIL_FILE.exists() else 1
        FAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
        FAIL_FILE.write_text(str(n))
        return None
    FAIL_FILE.unlink(missing_ok=True)
    return total


def costlog_total() -> float:
    """Sum the 'est $' column (5th) of the COSTLOG.md table."""
    if not COSTLOG.exists():
        return 0.0
    total = 0.0
    for line in COSTLOG.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and re.fullmatch(r"\$?\d+(\.\d+)?", cells[4]):
            total += float(cells[4].lstrip("$"))
    return total


def check() -> tuple[int, float | None, float]:
    billed = billing_month_to_date()
    logged = costlog_total()
    spent = max(billed or 0.0, logged)
    print(f"[cost_guard] billing month-to-date=${billed if billed is not None else float('nan'):.2f} "
          f"costlog estimate=${logged:.2f} -> using ${spent:.2f} (warn ${WARN_USD}, stop ${STOP_USD})")
    if spent >= STOP_USD:
        print("[cost_guard] HARD STOP: budget reached. No more GPU work.")
        return 2, billed, logged
    if billed is None and FAIL_FILE.exists() and int(FAIL_FILE.read_text()) >= MAX_FAILS:
        print("[cost_guard] billing failed repeatedly; check https://modal.com/settings/usage before continuing.")
        return 3, billed, logged
    if spent >= WARN_USD:
        print("[cost_guard] WARNING: above the warn threshold.")
    return 0, billed, logged


if __name__ == "__main__":
    sys.exit(check()[0])
