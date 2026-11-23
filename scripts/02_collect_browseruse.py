#!/usr/bin/env python3
"""02_collect_browseruse — funnel-crawl engine built on the browser-use
library's Agent, as a parallel alternative to the (unbuilt) native
Qwen3-VL custom-loop `02_collect.py`. Both write into the SAME
observations/site_outcomes tables, distinguished only by nav_method
(`browseruse_mlx` here), specifically so reach-rate and downstream
detection outcomes are directly comparable over the same drawn sample.

BUILD STATUS: build-order steps 2-3 done (see
~/.claude/plans/0-your-first-replicated-island.md). The preflight check,
--dry-run, and a single hardcoded site end-to-end (--test-site-id) are all
implemented and verified LIVE against a real site (vape.bg) across 10
live-test iterations — real age-gate dismissal, a full funnel walk to a
genuine cart page, correct DB writes, correct debug trail. See CLAUDE.md's
"MLX / browser-use gotchas" section for the concrete bugs found and fixed
along the way (most importantly: BrowserProfile(keep_alive=True) is
required or Agent.close() kills the shared browser session after every
single checkpoint). Steps 4-5 (the chunk-aware site loop consuming
chunk_work.py --preset drawn-sample, --retry-blocked, restart cadence,
kill/resume verification) are NOT started — every real run so far is
exactly one hardcoded site per invocation.

Model: a Qwen3-VL-30B-A3B-Instruct fine-tune (`bu-30b-a3b-preview`, resolved
to `mlx-community/bu-30b-a3b-preview-4bit`) served locally via MLX — NOT
available on Ollama, which is why this script needs a second local
inference backend. qwen3.6:27b (relevance) and qwen3-vl:30b (detection, and
any future native-loop navigation) are UNCHANGED and stay on Ollama — both
are frozen, validated instruments; moving either to a different backend
would be creating a new instrument requiring fresh validation, for zero
benefit here. See config/global.yaml's `navigation_bu` block and CLAUDE.md
for the exact MLX server command — live-verified, not just documented.

CHECKPOINTS: site_arrival, store, product, add_to_cart, cart — pulled from
config/<name>.yaml (NOT a hardcoded module constant), sliced to everything
BEFORE the terminal checkpoint (checkout_initiation), which is never
loaded and never turned into an agent task. `cart` is this script's own
terminal stop, one step more conservative than the config's actual
terminal checkpoint. See _CHECKPOINT_ASSERTIONS below for the structural
check that enforces this at every load.

INDUSTRY NAMESPACES: sites.industry (a DB value, e.g. "nicotine") and
config/<name>.yaml's filename (e.g. "vape") are two different namespaces
that happen to collide for alcohol/gambling but not nicotine/vape —
common.config.config_name_for_sites_industry() bridges them. --industry
takes the DB value.

RUN_ID: deliberately has NO default and must be passed explicitly, reused
across every invocation of the same logical crawl wave. Unlike 01b's scope
(a stable prompt hash), this script's chunk_runs scope IS run_id — a fresh
auto-generated run_id every invocation would silently make site_outcomes-
based resume meaningless (every invocation would see 100% pending
regardless of prior progress) without ever raising an error. Pass the same
--run-id every time you resume this wave's crawl.

Run:
  ./.venv/bin/python scripts/02_collect_browseruse.py --industry nicotine --dry-run
  ./.venv/bin/python scripts/02_collect_browseruse.py --industry nicotine --run-id wave1 \\
      --chunk-set drawn-sample-v1 --chunk 0 --dry-run
  # Single hardcoded site, real crawl (build-order step 3 — works today):
  ./.venv/bin/python scripts/02_collect_browseruse.py --industry nicotine \\
      --run-id my-test-run --test-site-id vape_bg_4d3956
  # Chunk-aware site loop (build-order step 4) — NOT implemented yet.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

# MUST be set before `import browser_use` anywhere below — browser-use phones
# home via PostHog by default (a real, confirmed core dependency, not a
# rumor: `posthog` is unconditionally installed alongside it). This is a
# module-load-time env var, not a runtime call, so it has to land before the
# first `from browser_use import ...` line in this file or any module it
# imports transitively.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config, db  # noqa: E402
from common.config import config_name_for_sites_industry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSUMER = "02_collect"
NAV_METHOD = "browseruse_mlx"

# Per-checkpoint "how do you know you're done" framing, layered onto the
# GOAL text that lives centrally in config/<name>.yaml (single source of
# truth, shared with whatever the native-loop engine eventually uses). The
# cart entry's checkout/payment disclaimer is belt-and-braces ONLY — the
# real invariant is that no checkout task is ever constructed at all (see
# load_checkpoints() below), not this sentence.
_DONE_CONDITIONS = {
    "store": "Call done as soon as a product listing page is visible.",
    "product": "Call done once the product detail page has loaded.",
    "add_to_cart": "Call done once the add-to-cart action is confirmed.",
    "cart": (
        "Call done once the cart/basket page is visible. Do not proceed to "
        "checkout, payment, or order confirmation."
    ),
}


def load_checkpoints(industry: str) -> list[dict]:
    """site_arrival..cart, in order, sliced from config/<name>.yaml BEFORE
    the terminal checkpoint (checkout_initiation) — asserts the terminal
    checkpoint never leaks into the returned list and that the list's last
    entry is 'cart', failing loudly if the industry config's checkpoint
    order or terminal marking ever changes upstream. `industry` is the
    sites.industry DB value (e.g. "nicotine"), mapped internally to the
    config filename (e.g. "vape").
    """
    config_name = config_name_for_sites_industry(industry)
    all_checkpoints = config.checkpoints(config_name)
    terminal_names = set(config.terminal_checkpoints(config_name))

    checkpoints = [c for c in all_checkpoints if c["name"] not in terminal_names]

    assert checkpoints, f"no non-terminal checkpoints found for config/{config_name}.yaml"
    assert not (terminal_names & {c["name"] for c in checkpoints}), (
        f"a terminal checkpoint leaked into the task list for config/{config_name}.yaml — "
        "this script must never construct an agent task for a terminal checkpoint"
    )
    assert checkpoints[-1]["name"] == "cart", (
        f"expected the last non-terminal checkpoint to be 'cart', got "
        f"{checkpoints[-1]['name']!r} — config/{config_name}.yaml's checkpoint order "
        "changed upstream; this script's terminal-safety assumption (never even "
        "construct a task past 'cart') needs re-verifying before proceeding."
    )
    return checkpoints


def build_task(checkpoint: dict) -> str:
    """Agent task string = the config's GOAL (shared, centralized) + this
    script's own "how to know you're done" framing (local — the custom loop
    phrases done-ness differently, via a JSON action field rather than a
    tool call)."""
    suffix = _DONE_CONDITIONS.get(checkpoint["name"], "")
    return f"{checkpoint['goal']} {suffix}".strip()


# --------------------------------------------------------------------------- #
# Single-site crawl (build-order step 3). Fully async — browser-use's own
# architecture is async throughout, and a sync Playwright call cannot be
# made from inside an already-running asyncio event loop.
# --------------------------------------------------------------------------- #
async def _dump_debug_trail(history, run_id: str, site_id: str, checkpoint_name: str) -> None:
    """Per-agent-step screenshots + action record for one checkpoint's
    Agent.run(). NEVER read by the classification pass — kept under a
    distinct debug/ subtree (common.paths.debug_dir) precisely so that
    separation is structural, not just a convention. Screenshots are
    base64-decoded from history.screenshots() (browser-use's own record of
    what it saw at each step), not copied from wherever it may have
    internally cached them — this is the raw material a failure can be
    reconstructed from without a re-crawl.
    """
    from common import paths

    d = paths.debug_dir(run_id, site_id, checkpoint_name)
    d.mkdir(parents=True, exist_ok=True)

    for i, b64 in enumerate(history.screenshots(return_none_if_not_screenshot=True), start=1):
        if b64 is None:
            continue
        paths.debug_step_screenshot_path(run_id, site_id, checkpoint_name, i).write_bytes(
            base64.b64decode(b64)
        )

    trail = {
        "checkpoint": checkpoint_name,
        "number_of_steps": history.number_of_steps(),
        "is_done": history.is_done(),
        "is_successful": history.is_successful(),
        "urls": history.urls(),
        "action_names": history.action_names(),
        "errors": history.errors(),
        "total_duration_seconds": history.total_duration_seconds(),
        "final_result": history.final_result(),
    }
    paths.debug_trail_json_path(run_id, site_id, checkpoint_name).write_text(
        json.dumps(trail, indent=2, default=str)
    )


async def crawl_one_site(*, url: str, cctld: str, site_id: str, run_id: str,
                         checkpoints: list[dict], bu_cfg: dict, db_path: Path) -> None:
    """Walk one site through `checkpoints` end to end, using a SEPARATE
    Playwright client connected via CDP to browser-use's own browser
    (BrowserSession.cdp_url) for canonical screenshots and stealth
    injection — browser-use's own page object is CDP-native, not
    Playwright's, so common/screenshot.py cannot be called on it directly
    (see CLAUDE.md's MLX / browser-use gotchas). One Agent instance per
    checkpoint transition, run to completion, then discarded — never two
    agents alive at once, matching the checkpoint-scoped sub-agent
    architecture. Writes observations/site_outcomes/runs directly; this is
    step 3 of the build order (single hardcoded site, no chunking, no site
    loop) — the real chunk-aware site loop is step 4.
    """
    from browser_use import Agent, BrowserSession, ChatOpenAI
    from browser_use.browser.profile import BrowserProfile
    from playwright_stealth import Stealth

    from common import stealth as st
    from common import paths

    profile = st.profile_for_cctld(cctld)
    llm = ChatOpenAI(
        base_url=bu_cfg["llm_base_url"], model=bu_cfg["llm_model"],
        api_key="not-needed", timeout=bu_cfg["llm_timeout"],
        # DIAGNOSTIC (2026-08-23): default 4096 truncated the judge call;
        # 8192 STILL truncated a regular action step in the next live test.
        # This model's structured output (memory/eval/plan reasoning lives
        # INSIDE the regular JSON fields, not gated by use_thinking) varies
        # wildly in length step to step — some calls ~4s, others exceed
        # 8192 tokens. Raised further for headroom; not yet confirmed
        # sufficient.
        max_completion_tokens=16384,
    )

    browser_profile = BrowserProfile(
        headless=True,
        # ROOT CAUSE, found 2026-08-23 after 7 failed live-test iterations:
        # Agent.close() (called automatically at the end of every
        # Agent.run()) checks browser_profile.keep_alive — if falsy (the
        # default), it calls session.kill(), a FULL hard teardown of the
        # browser session, unconditionally, after every single checkpoint.
        # This is exactly why a Playwright CDP-bridge kept going stale/
        # ECONNREFUSED, and why even browser-use's OWN take_screenshot()
        # started raising "Root CDP client not initialized" after the first
        # checkpoint — it wasn't a bug in any capture strategy, the shared
        # session was being destroyed out from under all of them. True per
        # the original spec's core requirement (one BrowserSession, reused
        # across checkpoint sub-agents) only when this is explicitly True.
        keep_alive=True,
        # EXPLICIT — enable_default_extensions defaults to True, silently
        # injecting an ad-blocker (uBlock Origin), an auto-cookie-consent-
        # dismisser ("I still don't care about cookies"), and a URL cleaner
        # into every session. Confirmed live: the first test run's log
        # showed all three being downloaded and loaded unasked. This is a
        # real methodological risk for THIS project specifically — invariant
        # #5 requires screenshots to be the unmodified source of truth, and
        # some sites implement age gates with the same modal/banner pattern
        # as cookie-consent banners. An extension silently auto-dismissing
        # something before our screenshot is an invisible confound: the
        # image would simply show no gate, indistinguishable from a site
        # that never had one. Disabled outright rather than trusted to only
        # ever touch cookie banners.
        enable_default_extensions=False,
        # EXPLICIT — captcha_solver defaults to True in browser-use, which
        # would otherwise silently violate invariant #2 (never solve
        # CAPTCHAs, never configure the agent with any CAPTCHA-solving
        # capability). Confirmed via BrowserProfile.model_fields directly.
        captcha_solver=False,
        # No native locale=/timezone_id= exists on BrowserSession/
        # BrowserProfile (confirmed by inspecting all 62 constructor
        # params) — set via raw Chromium launch args / process env instead,
        # the lower-level equivalent of common.stealth.new_context()'s
        # Playwright-native locale=/timezone_id= context options.
        args=[f"--lang={profile['locale']}"],
        env={"TZ": profile["timezone_id"]},
    )
    session = BrowserSession(browser_profile=browser_profile)
    await session.start()

    conn = db.connect(db_path)
    db.init_db(conn)
    db.start_run(conn, run_id, notes="02_collect_browseruse single-site test (build-order step 3)")

    max_checkpoint_reached: Optional[str] = None
    terminal_status: Optional[str] = None
    drop_checkpoint: Optional[str] = None
    drop_reason: Optional[str] = None

    stealth_js = Stealth().script_payload  # raw JS, for evaluate() injection — see inject_stealth() below

    async def inject_stealth() -> None:
        """Best-effort fingerprint hardening via browser-use's OWN page
        object (actor.page.Page.evaluate()), not a second Playwright
        connection. ABANDONED (2026-08-23) after repeated live failures: a
        CDP-bridged Playwright client connected once at the start of a site
        visit went stale after the first checkpoint's Agent.run() (its
        internal reset() disrupts the underlying CDP endpoint enough that
        even a freshly-cached cdp_url raised ECONNREFUSED on the second
        checkpoint) — too fragile to keep chasing.

        Known limitation, real and unresolved: playwright-stealth's payload
        is designed for INIT-SCRIPT injection (run before any page JS
        executes); evaluate() runs it AFTER the page has already loaded, so
        it cannot undo whatever the page's own scripts already read/set on
        first load. It may still help for properties fingerprinting scripts
        re-check dynamically, but this is not equivalent to real init-script
        stealth. A residual, documented gap — not silently claimed as fixed.
        """
        page = await session.get_current_page()
        if page is None:
            return
        try:
            await page.evaluate(stealth_js)
        except Exception:  # noqa: BLE001 — best-effort; must never break the crawl
            pass

    async def capture_checkpoint(checkpoint_name: str) -> None:
        """Canonical screenshot via BrowserSession.take_screenshot() — NOT a
        second CDP-bridged Playwright connection (abandoned, see
        inject_stealth() docstring for why). take_screenshot() is the SAME
        primitive the agent's own debug-trail screenshots already use
        reliably across every live test so far, so it survives whatever
        reset() does between checkpoints. common/screenshot.py's Playwright-
        `page.screenshot()`-based functions cannot be reused for this engine
        at all — confirmed, not just suspected.
        """
        fp = paths.fullpage_path(run_id, site_id, checkpoint_name)
        vp = paths.viewport_path(run_id, site_id, checkpoint_name)
        fp.parent.mkdir(parents=True, exist_ok=True)
        vp.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.sleep(1.5)  # short settle for late client-side rendering
        await session.take_screenshot(path=str(fp), full_page=True)
        await session.take_screenshot(path=str(vp), full_page=False)

    try:
        for checkpoint in checkpoints:
            name = checkpoint["name"]
            nav_steps = 0

            if name == "site_arrival":
                await session.navigate_to(url)
                await inject_stealth()
            else:
                # One Agent instance per checkpoint transition — cold start
                # on a warm browser: it inherits the physical page state
                # left by the previous agent/direct-nav, but carries zero
                # message history and no awareness that other agents exist.
                agent = Agent(
                    task=build_task(checkpoint),
                    llm=llm,
                    browser_session=session,
                    use_vision=True,  # explicit True, not "auto" — vision must be on every step unconditionally for a measurement instrument
                    vision_detail_level="high",
                    max_actions_per_step=1,
                    generate_gif=False,
                    calculate_cost=False,
                    step_timeout=bu_cfg["step_timeout"],
                    llm_timeout=bu_cfg["llm_timeout"],
                    # DIAGNOSTIC (2026-08-23): every step timed out at 60s in
                    # the first live test against bu-30b-a3b-preview, despite
                    # a direct isolated vision call against the SAME model/
                    # server with a REAL page screenshot completing in 4.3s —
                    # so raw inference speed isn't the bottleneck. use_thinking
                    # defaults to True; disabling it isolates whether
                    # chain-of-thought generation on a reasoning-tuned model
                    # is what's blowing past the timeout. Revisit once this
                    # is confirmed either way — do not treat as a permanent
                    # decision yet.
                    use_thinking=False,
                )
                # max_steps belongs on .run(), NOT the Agent() constructor —
                # confirmed via inspect.signature; the constructor has a
                # catch-all **kwargs that would otherwise silently swallow
                # it and let the agent run to browser-use's real default of
                # 500 steps per checkpoint.
                history = await agent.run(max_steps=bu_cfg["max_steps_per_checkpoint"])
                await _dump_debug_trail(history, run_id, site_id, name)
                nav_steps = history.number_of_steps()

                # BUG FOUND LIVE (2026-08-23): is_done() alone is not
                # enough. It's only "did the agent call done() at all" (vs.
                # exhausting max_steps) — it says nothing about whether the
                # agent's OWN done(success=...) call reported success or
                # failure. Confirmed live: the agent gave up on the `cart`
                # checkpoint after repeatedly failing to click a cart icon,
                # explicitly called done(success=False) admitting it never
                # left the product page, and this line let it through as
                # reached=1/"completed" — the checkpoint's own recorded URLs
                # showed all 6 steps stuck on the SAME product page.
                # is_successful() reads the agent's own success parameter
                # directly (browser_use/agent/views.py), not the separate,
                # frequently-truncating judge mechanism — it's guaranteed
                # non-None whenever is_done() is True.
                if not (history.is_done() and history.is_successful()):
                    drop_checkpoint = name
                    drop_reason = "nav_stuck"
                    db.upsert_observation(
                        conn, run_id=run_id, site_id=site_id, checkpoint=name,
                        reached=0, nav_method=NAV_METHOD, nav_steps=nav_steps,
                        failure_reason="nav_stuck",
                    )
                    break

            # The script takes the canonical screenshot, never the agent —
            # control is back in Python here. Re-inject stealth defensively
            # before capturing (best-effort; see inject_stealth()'s
            # docstring for its real limitations).
            await inject_stealth()
            await capture_checkpoint(name)
            fp = paths.fullpage_path(run_id, site_id, name)
            vp = paths.viewport_path(run_id, site_id, name)

            db.upsert_observation(
                conn, run_id=run_id, site_id=site_id, checkpoint=name, reached=1,
                fullpage_path=str(fp), viewport_path=str(vp),
                nav_method=NAV_METHOD, nav_steps=nav_steps,
            )
            max_checkpoint_reached = name

        if drop_checkpoint is None:
            terminal_status = "completed" if max_checkpoint_reached == checkpoints[-1]["name"] else "error"
        else:
            terminal_status = "dropped_stuck"

    except Exception as e:  # noqa: BLE001 — must still write a site_outcomes row before propagating
        drop_reason = f"{type(e).__name__}: {e}"
        terminal_status = "error"
        raise
    finally:
        # kill(), not stop() — with keep_alive=True (needed so Agent.close()
        # doesn't tear down the session between checkpoints), stop() alone
        # "keeps the browser alive" per its own docstring, which would leak
        # a Chromium process per site once a real site loop exists. kill()
        # is the deliberate, full teardown once the WHOLE visit is done.
        await session.kill()

        db.upsert_site_outcome(
            conn, run_id=run_id, site_id=site_id,
            max_checkpoint_reached=max_checkpoint_reached, terminal_status=terminal_status,
            drop_checkpoint=drop_checkpoint, drop_reason=drop_reason,
        )
        db.finish_run(conn, run_id)
        conn.close()
        print(f"\nsite_outcome: max_checkpoint_reached={max_checkpoint_reached!r} "
              f"terminal_status={terminal_status!r} drop_checkpoint={drop_checkpoint!r} "
              f"drop_reason={drop_reason!r}")


def _tiny_png_bytes() -> bytes:
    """A minimal, valid PNG for the preflight vision check. Content doesn't
    matter — only that SOME image round-trips through the endpoint and the
    model replies with something. Built with Pillow (already a project
    dependency) rather than an embedded base64 blob, so it's inspectable
    and regenerable rather than opaque.
    """
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _preflight_vision_request(base_url: str, model: str, timeout_s: int) -> str:
    """The actual round-trip: one vision request through the SAME ChatOpenAI
    client class the real agents will use. Returns the raw completion text.
    Never catches exceptions here — the caller decides how loudly to fail.
    """
    from browser_use.llm import ChatOpenAI, ContentImage, ContentText, UserMessage
    from browser_use.llm.messages import ImageURL

    llm = ChatOpenAI(base_url=base_url, model=model, api_key="not-needed", timeout=timeout_s)
    b64 = base64.b64encode(_tiny_png_bytes()).decode("ascii")
    message = UserMessage(
        content=[
            ContentText(text="Reply with one word describing this image."),
            ContentImage(image_url=ImageURL(url=f"data:image/png;base64,{b64}", detail="high", media_type="image/png")),
        ]
    )
    result = await llm.ainvoke([message])
    return (result.completion or "").strip()


def preflight_vision_check(base_url: str, model: str, timeout_s: int) -> str:
    """Synchronous wrapper. Aborts the WHOLE run loudly on any failure — a
    silent fallback to text-only would quietly invalidate every observation
    downstream: MLX's plain mlx_lm.server has no vision support at all but
    would still return SOME text for a text-only request, which could look
    superficially fine while never having actually seen an image. This
    check is what stands between that failure mode and a wasted multi-hour
    crawl, so it must run before any site is touched, not be skippable.
    """
    if not model:
        raise SystemExit(
            "ERROR: config/global.yaml navigation_bu.llm_model is empty — set it to "
            "the same identifier passed to the MLX server's --model flag. See CLAUDE.md "
            "for the exact server command."
        )
    print(f"preflight — base_url={base_url} model={model!r}")
    try:
        text = asyncio.run(_preflight_vision_request(base_url, model, timeout_s))
    except Exception as e:  # noqa: BLE001 — deliberately broad: ANY failure here must abort loudly
        raise SystemExit(
            f"ERROR: preflight vision check failed — {type(e).__name__}: {e}\n"
            f"  base_url={base_url} model={model!r}\n"
            "  Is the MLX vision server actually running and reachable? (mlx_lm.server "
            "is text-only and cannot serve this — see CLAUDE.md for the mlx-vlm command.)"
        ) from e
    if not text:
        raise SystemExit(
            f"ERROR: preflight vision check got an EMPTY response from base_url={base_url} "
            f"model={model!r} — the endpoint responded but returned nothing, which is exactly "
            "the silent-degradation failure mode this check exists to catch."
        )
    print(f"preflight OK — model replied: {text!r}")
    return text


def _resolve_chunk_index(conn, chunk_set_id: str, chunk_arg: str, *, run_id: str,
                         industry: Optional[str], retry_blocked: bool) -> Optional[int]:
    if chunk_arg == "next":
        return db.next_pending_collect_chunk_index(
            conn, chunk_set_id, run_id=run_id, industry=industry, retry_blocked=retry_blocked,
        )
    try:
        return int(chunk_arg)
    except ValueError:
        raise SystemExit(f"ERROR: --chunk must be an integer or 'next', got {chunk_arg!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=db.DEFAULT_DB)
    ap.add_argument("--industry", required=True, choices=["nicotine", "alcohol", "gambling"],
                    help="sites.industry DB value — NOT the config/<name>.yaml filename")
    ap.add_argument("--run-id", dest="run_id", default=None,
                    help="REQUIRED for any real (non-preflight-only) invocation — no default; "
                         "see module docstring for why auto-generating one is unsafe here")
    ap.add_argument("--chunk-set", dest="chunk_set", default=None,
                    help="chunk set id from `chunk_work.py create --preset drawn-sample --consumer 02_collect`")
    ap.add_argument("--chunk", default=None, help="chunk index within --chunk-set, or 'next'")
    ap.add_argument("--limit", type=int, default=None, help="cap sites resolved this invocation, for smoke tests")
    ap.add_argument("--retry-blocked", action="store_true",
                    help="also re-queue sites whose prior outcome for this run was dropped_blocked")
    ap.add_argument("--dry-run", action="store_true", help="preflight + resolve the site plan, no browsing")
    ap.add_argument("--test-site-id", dest="test_site_id", default=None,
                    help="build-order step 3: crawl ONE hardcoded site_id (from `sites`) end-to-end, "
                         "bypassing chunking entirely. Incompatible with --chunk-set/--dry-run.")
    args = ap.parse_args()

    if (args.chunk_set is None) != (args.chunk is None):
        ap.error("--chunk-set and --chunk must be given together")
    if args.test_site_id is not None and (args.chunk_set is not None or args.dry_run):
        ap.error("--test-site-id cannot be combined with --chunk-set or --dry-run")
    if args.test_site_id is not None and args.run_id is None:
        ap.error("--run-id is required with --test-site-id (see module docstring)")

    checkpoints = load_checkpoints(args.industry)
    print(f"02_collect_browseruse — industry={args.industry!r} "
          f"config={config_name_for_sites_industry(args.industry)!r} "
          f"checkpoints={[c['name'] for c in checkpoints]}")
    for c in checkpoints:
        if c["name"] == "site_arrival":
            print(f"  {c['name']:12} -> direct navigation, no agent")
        else:
            print(f"  {c['name']:12} -> {build_task(c)!r}")

    bu_cfg = config.load_global()["navigation_bu"]
    preflight_vision_check(bu_cfg["llm_base_url"], bu_cfg["llm_model"], bu_cfg["llm_timeout"])

    if args.test_site_id is not None:
        conn = db.connect(args.db)
        db.init_db(conn)
        site = db.get_site(conn, args.test_site_id)
        conn.close()
        if site is None:
            raise SystemExit(f"ERROR: no such site_id {args.test_site_id!r} in {args.db}")
        print(f"\n[STEP 3 SINGLE-SITE TEST] site_id={site['site_id']!r} url={site['url_normalized']!r} "
              f"cctld={site['cctld']!r} run_id={args.run_id!r}")
        asyncio.run(crawl_one_site(
            url=site["url_normalized"], cctld=site["cctld"], site_id=site["site_id"],
            run_id=args.run_id, checkpoints=checkpoints, bu_cfg=bu_cfg, db_path=args.db,
        ))
        return

    if args.chunk_set is not None:
        if args.run_id is None:
            ap.error("--run-id is required when --chunk-set/--chunk are given (see module docstring)")

        conn = db.connect(args.db)
        db.init_db(conn)

        chunk_set = db.get_chunk_set(conn, args.chunk_set)
        if chunk_set is None:
            raise SystemExit(
                f"ERROR: no such chunk set {args.chunk_set!r} — create it with "
                "scripts/chunk_work.py create --preset drawn-sample --consumer 02_collect"
            )

        chunk_index = _resolve_chunk_index(
            conn, args.chunk_set, args.chunk, run_id=args.run_id,
            industry=args.industry, retry_blocked=args.retry_blocked,
        )
        if chunk_index is None:
            print(f"\nchunk set {args.chunk_set!r}: every chunk is already fully processed "
                  f"for run_id={args.run_id!r}. Nothing to do.")
            conn.close()
            return

        pending = list(db.iter_pending_collect(
            conn, run_id=args.run_id, industry=args.industry, retry_blocked=args.retry_blocked,
            chunk_set=args.chunk_set, chunk_index=chunk_index,
        ))
        n_members = len(list(db.iter_chunk_members(conn, args.chunk_set, chunk_index)))
        if args.limit is not None:
            pending = pending[: args.limit]

        print(f"\nrun_id={args.run_id!r} chunk_set={args.chunk_set!r} chunk={chunk_index} "
              f"n_members={n_members} pending_this_invocation={len(pending)}"
              f"{'  [retry_blocked]' if args.retry_blocked else ''}")
        for site in pending[:20]:
            print(f"  {site['site_id']:40} {site['url_normalized']}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")

        conn.close()
    else:
        print("\n(no --chunk-set given — preflight only, no site plan resolved)")

    if args.dry_run:
        print("\n[DRY RUN — no browser opened, nothing written]")
        return

    raise SystemExit(
        "The chunk-aware site loop (build-order step 4) is not implemented yet — "
        "only --dry-run and --test-site-id (a single hardcoded site, step 3) actually crawl. "
        "See ~/.claude/plans/0-your-first-replicated-island.md for the staged build order."
    )


if __name__ == "__main__":
    main()
