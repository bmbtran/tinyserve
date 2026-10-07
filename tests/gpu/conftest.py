"""GPU suites: every test adds numbers to REPORT; at session end the report is
written to $TS_RESULT_JSON (returned to the laptop by modal_app.gpu_tests)
with status PASS only if every test passed."""

import json
import os
import time

import pytest

REPORT: dict = {"metrics": {}, "tests": {}}


@pytest.fixture(scope="session")
def report():
    return REPORT["metrics"]


def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        REPORT["tests"][report.nodeid.split("::")[-1]] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("TS_RESULT_JSON")
    status = "PASS" if exitstatus == 0 else "FAIL"
    REPORT["status"] = status
    REPORT["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    REPORT["git_sha"] = os.environ.get("TS_GIT_SHA")
    try:
        import subprocess

        REPORT["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                       capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    suite = os.path.basename(path or "suite").split(".")[0].upper()
    metrics = " ".join(f"{k}={v}" for k, v in REPORT["metrics"].items() if not isinstance(v, (list, dict)))
    print(f"\n{suite} {status} {metrics}")
    if path:
        with open(path, "w") as f:
            json.dump(REPORT, f, indent=2, default=str)
