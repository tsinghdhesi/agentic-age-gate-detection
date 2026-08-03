#!/usr/bin/env python3
"""Day-1 grounding spike — does Qwen3-VL click-grounding + the 1000x1000 rescale
actually work before we trust the loop?

For each URL: open it, capture the VIEWPORT, ask the nav model to locate an
obvious target (the shop/store/menu link), rescale the predicted box CENTER to
pixels via common.coords, and DRAW the predicted click point onto the screenshot.
You eyeball the annotated PNGs in spike/out/grounding/.

Two modes:
  --mode raw  (default)  : raw visual grounding (the real pipeline approach).
  --mode som             : Set-of-Mark STUB — overlay numbered boxes on
                           Playwright-extracted clickables, ask the model to pick
                           a number. Switch to this if raw grounding is poor.

Standalone + throwaway. Run:  ./.venv/bin/python spike/spike_grounding.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

from common import config, stealth, vlm  # noqa: E402
from common.coords import box_center_to_pixel, model_to_pixel  # noqa: E402
import ollama  # noqa: E402

# ~10 candidate retailers sampled from the nicotine seed (some may be dead/parked
# — that's fine, the spike tolerates failures). Edit freely; it's throwaway.
URLS = [
    "https://puffbar.com",
    "https://www.vkvape.com",
    "https://elfbarvapes.de",
    "https://e-vaping.co.uk",
    "https://smokeland.it",
    "https://nzovape.com/",
    "https://atoresbrasileirosnus.blogspot.com",
    "https://no.vivivapes.com/",
    "https://spkrasiczyn.edupage.org",
    "https://esmoker.bg/",
]

GOAL = (
    "Find and click the primary navigation link that leads to the product shop "
    "or store (for example 'Shop', 'Store', 'Products', 'Vape', or the main menu)."
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "grounding")


def _host(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0]


def _cctld_guess(url: str) -> str:
    return _host(url).rsplit(".", 1)[-1]


def _slug(url: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in _host(url))


def _annotate(img_path, *, box_px=None, click_px=None, label="", out_path=None):
    """Draw the predicted box + click crosshair + a label banner onto the image."""
    img = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(img)
    if box_px is not None:
        d.rectangle(box_px, outline=(255, 0, 0), width=3)
    if click_px is not None:
        x, y = click_px
        r = 10
        d.line([(x - 18, y), (x + 18, y)], fill=(0, 200, 255), width=3)
        d.line([(x, y - 18), (x, y + 18)], fill=(0, 200, 255), width=3)
        d.ellipse([x - r, y - r, x + r, y + r], outline=(0, 200, 255), width=3)
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((4, 5), label[: img.width // 7], fill=(255, 255, 0))
    out = out_path or img_path
    img.save(out)
    return out


def run_raw(page, vp, out_png):
    """Real-pipeline grounding: ask nav model, rescale box center, annotate."""
    res = vlm.decide_navigation(out_png, GOAL)
    box, click_px, box_px = res.get("box"), None, None
    if box is not None:
        click_px = box_center_to_pixel(box, vp["width"], vp["height"])
        x1, y1 = model_to_pixel(box[0], box[1], vp["width"], vp["height"])
        x2, y2 = model_to_pixel(box[2], box[3], vp["width"], vp["height"])
        box_px = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    label = f"[raw] action={res['action']} click={tuple(round(c) for c in click_px) if click_px else None} {res['raw'][:60]}"
    _annotate(out_png, box_px=box_px, click_px=click_px, label=label)
    return res["action"], click_px


def run_som(page, vp, out_png, model):
    """Set-of-Mark STUB: number the clickables, ask the model to choose one."""
    els = page.evaluate(
        """() => {
            const sel = 'a,button,[role=button],input[type=submit],[onclick]';
            const out = [];
            for (const el of document.querySelectorAll(sel)) {
                const r = el.getBoundingClientRect();
                if (r.width < 8 || r.height < 8) continue;
                if (r.bottom < 0 || r.right < 0 ||
                    r.top > window.innerHeight || r.left > window.innerWidth) continue;
                out.push({x:r.x, y:r.y, w:r.width, h:r.height,
                          text:(el.innerText||el.value||'').trim().slice(0,40)});
            }
            return out.slice(0, 40);
        }"""
    )
    img = Image.open(out_png).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, e in enumerate(els):
        d.rectangle([e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]],
                    outline=(255, 0, 0), width=2)
        d.rectangle([e["x"], e["y"] - 14, e["x"] + 22, e["y"]], fill=(255, 0, 0))
        d.text((e["x"] + 2, e["y"] - 13), str(i), fill=(255, 255, 255))
    marked = out_png.replace(".png", "_som.png")
    img.save(marked)

    prompt = (
        "The image has numbered red boxes over clickable elements. Reply with "
        "ONLY the integer number of the element that best matches this goal, or "
        f"-1 if none match.\nGOAL: {GOAL}"
    )
    resp = ollama.Client(host=config.load_global()["ollama"]["host"]).generate(
        model=model, prompt=prompt, images=[open(marked, "rb").read()],
        think=False, options={"temperature": 0},
    )
    raw = (resp.response if hasattr(resp, "response") else str(resp)).strip()
    try:
        pick = int("".join(ch for ch in raw if ch.isdigit() or ch == "-"))
    except ValueError:
        pick = -1
    if 0 <= pick < len(els):
        e = els[pick]
        cx, cy = e["x"] + e["w"] / 2, e["y"] + e["h"] / 2
        d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], outline=(0, 200, 255), width=4)
        img.save(marked)
    return f"pick={pick}", raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "som"], default="raw")
    ap.add_argument("--model", default=None, help="override nav model")
    ap.add_argument("--limit", type=int, default=len(URLS))
    ap.add_argument("--headful", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    vp = config.load_global()["viewport"]
    model = args.model or config.load_global()["models"]["navigation"]

    print(f"grounding spike — mode={args.mode} model={model} -> {OUT_DIR}\n")
    with stealth.stealth_playwright() as p:
        browser = stealth.launch_browser(p, headless=not args.headful)
        for url in URLS[: args.limit]:
            out_png = os.path.join(OUT_DIR, f"{_slug(url)}.png")
            ctx = stealth.new_context(browser, cctld=_cctld_guess(url))
            page = ctx.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2500)
                page.screenshot(path=out_png, full_page=False)
                if args.mode == "raw":
                    action, click = run_raw(page, vp, out_png)
                    print(f"  {url:42} action={action:5} click={tuple(round(c) for c in click) if click else None}")
                else:
                    info, raw = run_som(page, vp, out_png, model)
                    print(f"  {url:42} {info}  raw={raw[:40]}")
            except Exception as e:  # noqa: BLE001 — spike is best-effort
                print(f"  {url:42} ERROR {type(e).__name__}: {str(e)[:60]}")
            finally:
                ctx.close()
        browser.close()
    print(f"\nDone. Open the annotated PNGs in {OUT_DIR} and check the cyan dot lands on a real shop/menu link.")


if __name__ == "__main__":
    main()
