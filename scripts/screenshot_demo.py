"""Screenshot of the browser demo replaying an LSFB-ISOL example clip.

Serves app/web locally, opens it in headless Chromium (Playwright), replays example clips until
the requested gloss comes up and saves the page to paper/figures/demo.png and
reports/figures/demo.png.

    uv run python scripts/screenshot_demo.py [--gloss EQUIPE]
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

from local_server import serve

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "app" / "web"
OUT = [ROOT / "paper" / "figures" / "demo.png", ROOT / "reports" / "figures" / "demo.png"]


async def capture(port: int, gloss: str, n_clips: int = 40) -> None:
    """Replay examples until `gloss` is shown and correctly recognised, then screenshot."""
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader"])
        page = await b.new_page(viewport={"width": 1360, "height": 860}, device_scale_factor=2)
        await page.goto(f"http://127.0.0.1:{port}/?model=lsfb")
        ready = "document.getElementById('status').dataset.state === 'ready'"
        await page.wait_for_function(ready, timeout=180_000)
        for _ in range(n_clips):
            await page.click("#btn-example")
            done = "document.getElementById('top1-hint').textContent.startsWith('True sign')"
            await page.wait_for_function(done, timeout=60_000)
            hint = await page.text_content("#top1-hint") or ""
            if hint.startswith(f"True sign: {gloss} ✓"):
                break
            await page.click("#btn-clear")
        else:
            raise RuntimeError(f"{gloss} was not replayed and recognised in {n_clips} clips")
        await page.mouse.move(0, 0)
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUT[0]))
        await b.close()
    for dst in OUT[1:]:
        shutil.copyfile(OUT[0], dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gloss", default="EQUIPE")
    args = ap.parse_args()
    with serve(WEB) as port:
        asyncio.run(capture(port, args.gloss))
    print("saved", *OUT)


if __name__ == "__main__":
    main()
