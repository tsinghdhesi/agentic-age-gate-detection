#!/usr/bin/env python3
"""Navigation spike — the FIRST code in this repo that actually ACTS on a page.

spike_grounding.py only asked the model where it *would* click and drew a
crosshair; it never called page.mouse.click, never chained steps, never walked
a funnel. So the core assumption of the whole project — that Qwen3-VL can drive
a real storefront from home page toward checkout — has never been tested. This
spike tests it, and its findings determine how scripts/02_collect.py gets built.

Walks: site_arrival -> store -> product -> add_to_cart.
STOPS at add_to_cart. It deliberately never approaches `cart` or the terminal
`checkout_initiation`: invariant #1's structural guard is Phase-3 work that does
not exist yet, and a throwaway script has no business near a real checkout.

Questions this is meant to answer (record the answers, don't just eyeball
pass/fail — they are design inputs for 02_collect):
  1. Does the model pick sensible targets on real storefronts, or get captured
     by cookie banners and promo modals?
  2. Do AGE GATES block the crawl? The thing being measured is also an obstacle.
     Clicking through one is methodologically fine — detection is a downstream
     pass over stored screenshots (invariant #5), so the gate is recorded before
     it is dismissed — but the loop has to handle it deliberately.
  3. Is loop.step_cap (6) enough per checkpoint? Is no_progress_limit (2) right?
  4. Does raw grounding suffice, or is a Set-of-Mark fallback needed?

Standalone + throwaway. Run:
  ./.venv/bin/python spike/spike_navigation.py
  ./.venv/bin/python spike/spike_navigation.py --headful --limit 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config, screenshot, stealth, vlm  # noqa: E402
from common.coords import box_center_to_pixel  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "navigation")

# Known-good retailers straight from the hand-labeled gold set (label=1,
# confirmed reachable, deliberately multilingual). (url, cctld)
TARGETS = [
    ("https://vape.bg", "bg"),                    # BG — has its own age gate
    ("https://www.dutchvapeshop.nl", "nl"),       # NL — has its own age gate
    ("https://royalvape.cz", "cz"),
    ("http://www.vapourmate.co.uk", "co.uk"),
    ("https://e6-vapeshop.de", "de"),
    ("https://caktusvape.co.nz", "co.nz"),        # NZ — has its own age gate
    ("https://jatostovapershop.es", "es"),
    ("https://vaporhub.ae", "ae"),
]

# The funnel prefix this spike is willing to attempt. Everything from `cart`
# onward is deliberately excluded — see module docstring.
SPIKE_CHECKPOINTS = ["site_arrival", "store", "product", "add_to_cart"]


def _dom_hash(page) -> str:
    """Hash of the rendered DOM — the 'no progress' signal global.yaml's
    loop.no_progress_limit comment already assumes exists ("unchanged URL+DOM-hash
    for N actions -> stuck"). 02_collect will need this for real, and will also
    need to persist the DOM to observations.dom_path (currently unimplemented).
    """
    try:
        return hashlib.sha256(page.content().encode("utf-8", "replace")).hexdigest()[:12]
    except Exception:  # noqa: BLE001 — spike is best-effort
        return "?"


def _act(page, decision: dict, vp: dict) -> str:
    """Execute one model decision for real. Returns a short description.

    This is the part spike_grounding never did. Coordinates come back on the
    1000x1000 canvas and MUST be rescaled via common.coords before touching
    Playwright (CLAUDE.md's CRITICAL section).
    """
    action = decision["action"]
    box = decision.get("box")

    if action == "click":
        px, py = box_center_to_pixel(box, vp["width"], vp["height"])
        page.mouse.click(px, py)
        return f"click @ ({px:.0f},{py:.0f}) from box {box}"

    if action == "type":
        px, py = box_center_to_pixel(box, vp["width"], vp["height"])
        page.mouse.click(px, py)
        page.keyboard.type(decision.get("text") or "")
        return f"type {decision.get('text')!r} @ ({px:.0f},{py:.0f})"

    if action == "scroll":
        direction = (decision.get("text") or "down").strip().lower()
        dy = -600 if direction.startswith("up") else 600
        page.mouse.wheel(0, dy)
        return f"scroll {direction}"

    return action  # done | stuck — no page interaction


def run_checkpoint(page, goal: str, vp: dict, step_cap: int, no_progress_limit: int,
                   slug: str, checkpoint: str, model: str, log: list) -> str:
    """Drive one checkpoint to completion. Returns 'done' | 'stuck' | 'cap'."""
    no_progress = 0
    for step in range(1, step_cap + 1):
        shot = os.path.join(OUT_DIR, f"{slug}__{checkpoint}__s{step}.png")
        screenshot.viewport(page, shot)

        url_before, dom_before = page.url, _dom_hash(page)
        d = vlm.decide_navigation(shot, goal, model=model)

        entry = {
            "site": slug, "checkpoint": checkpoint, "step": step,
            "action": d["action"], "box": d.get("box"), "text": d.get("text"),
            "url_before": url_before, "dom_before": dom_before,
            "latency_ms": d["latency_ms"], "raw": d["raw"][:120],
        }

        if d["action"] in ("done", "stuck"):
            entry["executed"] = d["action"]
            entry["url_after"], entry["dom_after"] = url_before, dom_before
            entry["progressed"] = False
            log.append(entry)
            print(f"      s{step}: {d['action']}  ({d['latency_ms']}ms)")
            return d["action"]

        try:
            desc = _act(page, d, vp)
        except Exception as e:  # noqa: BLE001
            entry["executed"] = f"ERROR {type(e).__name__}"
            log.append(entry)
            print(f"      s{step}: action failed — {type(e).__name__}: {str(e)[:60]}")
            return "stuck"

        page.wait_for_timeout(2500)
        url_after, dom_after = page.url, _dom_hash(page)
        progressed = (url_after != url_before) or (dom_after != dom_before)

        entry.update(executed=desc, url_after=url_after, dom_after=dom_after,
                     progressed=progressed)
        log.append(entry)
        flag = "->" if progressed else "  (no change)"
        print(f"      s{step}: {desc}  ({d['latency_ms']}ms) {flag}")

        if progressed:
            no_progress = 0
        else:
            no_progress += 1
            if no_progress >= no_progress_limit:
                print(f"      no progress x{no_progress} — stuck")
                return "stuck"

    return "cap"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="override nav model")
    ap.add_argument("--limit", type=int, default=len(TARGETS))
    ap.add_argument("--headful", action="store_true")
    args = ap.parse_args()

    g = config.load_global()
    vp = g["viewport"]
    step_cap = g["loop"]["step_cap"]
    no_progress_limit = g["loop"]["no_progress_limit"]
    model = args.model or g["models"]["navigation"]

    goals = {c["name"]: c["goal"] for c in config.checkpoints("vape")}
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"navigation spike — model={model}  viewport={vp['width']}x{vp['height']}  "
          f"step_cap={step_cap}  no_progress_limit={no_progress_limit}")
    print(f"walking: {' -> '.join(SPIKE_CHECKPOINTS)}   (stops before cart/checkout)")
    print(f"output -> {OUT_DIR}\n")

    log: list = []
    depth: dict = {}
    t0 = time.time()

    with stealth.stealth_playwright() as p:
        browser = stealth.launch_browser(p, headless=not args.headful)
        for url, cctld in TARGETS[: args.limit]:
            slug = "".join(ch if ch.isalnum() else "_" for ch in url.split("//", 1)[-1])
            print(f"  {url}")
            ctx = stealth.new_context(browser, cctld=cctld)
            page = ctx.new_page()
            reached = []
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2500)
                screenshot.viewport(page, os.path.join(OUT_DIR, f"{slug}__site_arrival.png"))
                reached.append("site_arrival")
                print(f"    site_arrival: loaded ({page.url})")

                for cp in SPIKE_CHECKPOINTS[1:]:
                    print(f"    {cp}: goal={goals[cp]!r}")
                    outcome = run_checkpoint(page, goals[cp], vp, step_cap,
                                             no_progress_limit, slug, cp, model, log)
                    if outcome == "done":
                        reached.append(cp)
                        print(f"    {cp}: REACHED  ({page.url})")
                    else:
                        print(f"    {cp}: NOT reached ({outcome})")
                        break
            except Exception as e:  # noqa: BLE001
                print(f"    ERROR {type(e).__name__}: {str(e)[:70]}")
            finally:
                depth[url] = reached
                ctx.close()
        browser.close()

    with open(os.path.join(OUT_DIR, "nav_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, ensure_ascii=False)

    print(f"\n--- depth reached per site ({time.time()-t0:.0f}s) ---")
    for url, reached in depth.items():
        deepest = reached[-1] if reached else "(none)"
        print(f"  {url:42} {len(reached)}/{len(SPIKE_CHECKPOINTS)}  deepest={deepest}")

    counts = {cp: sum(1 for r in depth.values() if cp in r) for cp in SPIKE_CHECKPOINTS}
    print("\n--- sites reaching each checkpoint ---")
    for cp, n in counts.items():
        print(f"  {cp:18} {n}/{len(depth)}")

    acted = [e for e in log if e.get("progressed") is not None]
    if acted:
        prog = sum(1 for e in acted if e["progressed"])
        print(f"\nactions executed: {len(acted)}  progressed: {prog} ({100*prog/len(acted):.0f}%)")
    lat = [e["latency_ms"] for e in log if e.get("latency_ms")]
    if lat:
        lat.sort()
        print(f"nav latency: median {lat[len(lat)//2]}ms  max {lat[-1]}ms")
    print(f"\nfull step log -> {os.path.join(OUT_DIR, 'nav_log.json')}")


if __name__ == "__main__":
    main()
