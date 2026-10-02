"""End-to-end check that the browser demo predicts exactly like PyTorch.

Serves app/web locally, opens it in headless Chromium (Playwright), classifies every example clip
through the demo's own code path (JavaScript preprocessing + ONNX Runtime Web) and compares the
probabilities with the PyTorch checkpoint. Writes reports/browser_parity.json.

    uv run python scripts/check_browser_parity.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from playwright.async_api import async_playwright

from local_server import serve
from signrec.checkpoint import load_checkpoint
from signrec.preprocess import PreprocessConfig, preprocess

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "app" / "web"


def python_probs() -> tuple[list[np.ndarray], list[str], dict]:
    ex = json.loads((WEB / "examples/lsfb.json").read_text())
    rep = json.loads((WEB / "models/lsfb/export_report.json").read_text())
    model, ck = load_checkpoint(ROOT / "models" / rep["run"] / "best.pt")
    pc = PreprocessConfig.from_dict(ck["preprocess"])
    out = []
    for c in ex["clips"]:
        seq = np.array([np.nan if v is None else v for v in c["data"]], np.float32)
        x, m = preprocess(seq.reshape(c["T"], -1, 3), pc, y_scale=ex["y_scale"])
        with torch.no_grad():
            logits = model(torch.from_numpy(x[None]), torch.from_numpy(m[None]))
        out.append(logits.softmax(-1).numpy()[0])
    return out, ck["labels"], ex


async def browser_probs(port: int, n: int) -> list[dict]:
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader"])
        page = await b.new_page()
        await page.goto(f"http://127.0.0.1:{port}/?model=lsfb")
        ready = "document.getElementById('status').dataset.state === 'ready'"
        await page.wait_for_function(ready, timeout=180_000)
        res = [await page.evaluate(f"window.signrecExampleProbs({i})") for i in range(n)]
        await b.close()
    return res


def main() -> None:
    py, labels, ex = python_probs()
    with serve(WEB) as port:
        js = asyncio.run(browser_probs(port, len(py)))
    same = sum(int(np.argmax(a) == np.argmax(b["probs"])) for a, b in zip(py, js, strict=True))
    diff = max(float(np.abs(a - np.array(b["probs"])).max()) for a, b in zip(py, js, strict=True))
    acc = float(np.mean([labels[int(np.argmax(b["probs"]))] == b["label"] for b in js]))
    out = {
        "n_clips": len(js),
        "same_top1": same,
        "max_abs_prob_diff": diff,
        "top1_on_examples": acc,
    }
    (ROOT / "reports" / "browser_parity.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
