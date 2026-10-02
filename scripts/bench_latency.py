"""CPU latency of the exported models, natively (onnxruntime) and in a browser (onnxruntime-web).

    uv run python scripts/bench_latency.py \
        --ckpt asl-transformer=models/asl-transformer-s0/best.pt \
        --ckpt asl-gru=models/asl-gru-s0/best.pt

Writes reports/latency.json. The browser benchmark runs headless Chromium (Playwright), WebAssembly
backend, one thread, i.e. the configuration of the static demo (no cross-origin isolation).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

from local_server import serve
from signrec.checkpoint import load_checkpoint
from signrec.export import benchmark, export_onnx, quantize, session

ROOT = Path(__file__).resolve().parents[1]
ORT_WEB = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/"

PAGE = """<!doctype html><meta charset="utf-8">
<script type="module">
import * as ort from "__ORT__ort.min.mjs";
ort.env.wasm.wasmPaths = "__ORT__";
ort.env.wasm.numThreads = 1;
const sessions = {};
const times = {};
for (const [name, T, F] of __MODELS__) {
  const s = await ort.InferenceSession.create(name + ".onnx", { executionProviders: ["wasm"] });
  const x = new ort.Tensor("float32", new Float32Array(T * F).map(() => Math.random()), [1, T, F]);
  const m = new ort.Tensor("float32", new Float32Array(T).fill(1), [1, T]);
  for (let i = 0; i < 30; i++) await s.run({ x, mask: m });
  sessions[name] = { s, x, m };
  times[name] = [];
}
// interleave models over several rounds so that transient slow-downs affect all of them
for (let round = 0; round < 5; round++) {
  for (const [name, { s, x, m }] of Object.entries(sessions)) {
    for (let i = 0; i < 60; i++) {
      const t0 = performance.now();
      await s.run({ x, mask: m });
      times[name].push(performance.now() - t0);
    }
  }
}
const out = {};
for (const [name, t] of Object.entries(times)) {
  t.sort((a, b) => a - b);
  out[name] = { p50_ms: t[Math.floor(t.length * 0.5)], p95_ms: t[Math.floor(t.length * 0.95)] };
}
window.BENCH = out;
</script>"""


async def browser_bench(folder: Path, models: list[tuple[str, int, int]]) -> dict:
    page_src = PAGE.replace("__ORT__", ORT_WEB).replace("__MODELS__", json.dumps(models))
    (folder / "index.html").write_text(page_src)
    with serve(folder) as port:
        async with async_playwright() as p:
            b = await p.chromium.launch()
            page = await b.new_page()
            await page.goto(f"http://127.0.0.1:{port}/index.html")
            await page.wait_for_function("window.BENCH !== undefined", timeout=600_000)
            res = await page.evaluate("window.BENCH")
            await b.close()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path/to/best.pt")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    report, web_models = {}, []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        model, ck = load_checkpoint(ROOT / path)
        T, F = ck["preprocess"]["T"], ck["n_features"]
        fp32 = export_onnx(model, T, F, tmp / f"{name}_fp32.onnx")
        int8 = quantize(fp32, tmp / f"{name}_int8.onnx")
        for tag, f in (("fp32", fp32), ("int8", int8)):
            report[f"{name}_{tag}"] = {
                "size_mb": f.stat().st_size / 2**20,
                "native": benchmark(session(f), T, F, n=500),
            }
            web_models.append((f"{name}_{tag}", T, F))
    if not a.no_browser:
        web = asyncio.run(browser_bench(tmp, web_models))
        for k, v in web.items():
            report[k]["browser"] = v
    out = ROOT / "reports" / "latency.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
