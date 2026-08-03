#!/usr/bin/env python3
"""Day-1 detection spike — validates the FROZEN detection prompt and the
full_page-vs-viewport capture choice on a handful of pages.

Captures BOTH full_page and viewport for each URL, runs detect_age_gate() on
each, and prints a tiny confusion summary against PROVISIONAL labels. The labels
below are my best guess from the seed (real retailers -> likely gated; non-retail
noise -> ungated). EYEBALL the saved screenshots in spike/out/detection/ and fix
the labels here — that is the point of this spike.

Standalone + throwaway. Run:  ./.venv/bin/python spike/spike_detection.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config, screenshot, stealth, vlm  # noqa: E402
from common.prompts import DETECTION_PROMPT_HASH  # noqa: E402

# (url, provisional_label)  1 = age gate expected, 0 = not. PROVISIONAL — fix me.
LABELED = [
    ("https://puffbar.com", 0),
    ("https://www.vkvape.com", 1),
    ("https://elfbarvapes.de", 0),
    ("https://e-vaping.co.uk", 1),
    ("https://smokeland.it", 0),
    ("https://nzovape.com/", 1),
    ("https://atoresbrasileirosnus.blogspot.com", 0),
    ("https://no.vivivapes.com/", 0),
    ("https://spkrasiczyn.edupage.org", 0),
    ("https://esmoker.bg/", 0),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "detection")


def _host(url):
    return url.split("//", 1)[-1].split("/", 1)[0]


def _slug(url):
    return "".join(c if c.isalnum() else "_" for c in _host(url))


def _confusion(rows, key):
    """rows: list of dicts. key: 'fp' or 'vp'. Returns counts vs provisional labels."""
    tp = fp = tn = fn = unparsed = 0
    for r in rows:
        pred = r[key]
        if pred is None:
            unparsed += 1
            continue
        gold = r["label"]
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 1 and pred == 0:
            fn += 1
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, unparsed=unparsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="detection model (default 8b)")
    ap.add_argument("--limit", type=int, default=len(LABELED))
    ap.add_argument("--headful", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = args.model or config.load_global()["models"]["detection_default"]
    print(f"detection spike — model={model}  prompt_hash={DETECTION_PROMPT_HASH[:12]}")
    print(f"output -> {OUT_DIR}\n")
    print(f"{'url':42} {'gold':4} {'full':4} {'view':4}  raw(full)")

    rows = []
    with stealth.stealth_playwright() as p:
        browser = stealth.launch_browser(p, headless=not args.headful)
        for url, gold in LABELED[: args.limit]:
            slug = _slug(url)
            fp_png = os.path.join(OUT_DIR, f"{slug}_fullpage.png")
            vp_png = os.path.join(OUT_DIR, f"{slug}_viewport.png")
            ctx = stealth.new_context(browser, cctld=_host(url).rsplit(".", 1)[-1])
            page = ctx.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2500)
                screenshot.full_page(page, fp_png)
                screenshot.viewport(page, vp_png)
                fp = vlm.detect_age_gate(fp_png, model=model)
                vp = vlm.detect_age_gate(vp_png, model=model)
                rows.append({"url": url, "label": gold, "fp": fp["label"], "vp": vp["label"]})
                print(f"  {url:40} {gold!s:4} {str(fp['label']):4} {str(vp['label']):4}  {fp['raw'][:24]!r}")
            except Exception as e:  # noqa: BLE001 — spike is best-effort
                print(f"  {url:40} {gold!s:4} ERR  ERR   {type(e).__name__}: {str(e)[:40]}")
            finally:
                ctx.close()
        browser.close()

    print("\n--- confusion vs PROVISIONAL labels (fix LABELED after eyeballing) ---")
    for key, name in (("fp", "full_page"), ("vp", "viewport")):
        c = _confusion(rows, key)
        acc = (c["tp"] + c["tn"]) / max(1, c["tp"] + c["tn"] + c["fp"] + c["fn"])
        print(f"  {name:10} TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']} "
              f"unparsed={c['unparsed']}  acc≈{acc:.2f}")
    print("\nReminder: full_page is the default precisely to avoid below-the-fold "
          "false negatives. Compare FN counts between the two captures.")


if __name__ == "__main__":
    main()
