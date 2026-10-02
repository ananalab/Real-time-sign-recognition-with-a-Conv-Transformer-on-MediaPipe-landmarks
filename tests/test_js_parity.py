"""The browser preprocessing (app/web/js/preprocess.js) must match preprocess() exactly."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_preprocess_matches_python() -> None:
    subprocess.run([sys.executable, "scripts/export_web_assets.py"], cwd=ROOT, check=True)
    out = subprocess.run(
        ["node", "app/web/test/parity.mjs", "tests/fixtures/js_parity.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report = json.loads(out.stdout)
    failed = [r for r in report if not r["pass"]]
    assert not failed, failed
    assert len(report) >= 9


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_segmenter() -> None:
    subprocess.run(["node", "app/web/test/segmenter.mjs"], cwd=ROOT, check=True)
