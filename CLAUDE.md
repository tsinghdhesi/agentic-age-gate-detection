# CLAUDE.md — Agentic Age-Gate Detection Pipeline

Project context for future sessions. Keep this current as the build progresses.

## What this is
A research data-collection pipeline (CS thesis). It crawls e-cigarette /
alcohol / gambling sites through the purchase funnel
(`site_arrival → store → product → add_to_cart → cart → checkout_initiation`)
and, at each checkpoint, screenshots the page and detects whether an **age gate**
is present. **Unit of analysis = per-checkpoint age-gate detection.** Navigation
is instrumental plumbing, not a research output.

Runs **fully locally**: Qwen3-VL via **Ollama at http://localhost:11434**.

## Non-negotiable invariants (encoded structurally, not via prompts)
1. **Never complete a real purchase.** Checkpoints with `terminal: true` are a
   structural wall — the nav loop must be *incapable* of acting once a terminal
   checkpoint is reached (guard in code, Phase 3).
2. **Never solve CAPTCHAs.** A CAPTCHA → `blocked_bot` data point, site dropped.
3. **No cloud calls, ever.** All inference is local Ollama. No openai/anthropic/
   etc. imports anywhere (enforced in `requirements.txt` header too).
4. **Serial execution only.** One browser context, one site at a time. No
   parallelism/async pools, no future hooks for them.
5. **Screenshots are the source of truth.** Every checkpoint persists PNG(s).
   Detection is a separate downstream pass over those PNGs. `02_collect.py`
   MUST NOT write age-gate labels.
6. **The detection prompt is a frozen instrument.** Stored in ONE place, hashed,
   hash recorded per run (`runs.detection_prompt_hash`). Never edited inline.
   (The prompt module itself lands in Phase 2.)

## CRITICAL — Qwen3-VL coordinates
Qwen3-VL emits coords on a normalized **1000×1000 canvas** (NOT pixels, NOT
0–1 floats). Rescale before every Playwright action:
`actual_x = (model_x/1000)*viewport_w`, `actual_y = (model_y/1000)*viewport_h`.
This lives in exactly one place: [common/coords.py](common/coords.py), proven by
[tests/test_coords.py](tests/test_coords.py). Use `box_center_to_pixel()` in the
loop. Never re-derive the math inline.

## Locked design decisions
- **site_id = `<hostslug>_<6char url hash>`** (e.g. `m_hindi_vampedvape_com_3f9a1c`).
  Stable across resumes, debuggable on disk, collision-safe per distinct URL.
  Also the screenshot/dom folder name.
- **Queue carries inclusion columns.** `01_build_queue` ingests ALL seed URLs
  (normalize + dedupe, full provenance). A *separate later pass* sets
  `included` / `relevance_label` / `sample_stratum`. Crawler reads only
  `included = 1`. (Reason: the seed is 133,981 noisy CrUX URLs — far more than a
  serial crawl can cover.)
- **`cctld` stores the EXACT public suffix**, never null — `com`, `org`,
  `co.uk`, `com.br`, `ru`, etc. (user directive). Extraction in `01_build_queue`
  will use **tldextract configured OFFLINE** (bundled PSL snapshot, no network)
  to handle multi-label suffixes against the noisy seed.
- **`checkout_initiation` is the terminal checkpoint** (the no-purchase wall).
- **One table per concept, `industry` as a column** (set by `01_build_queue
  --industry`, default inferred from seed filename). Never fork the schema.
- **Capture strategy (do not flip):** age-gate detection = **full_page** default;
  navigation grounding = **viewport** always. Collection stores BOTH at every
  checkpoint, so `detection_capture` (in `global.yaml`) is a re-classification
  toggle, never a re-crawl. `navigation_capture` is fixed at viewport.

## Data model (SQLite, [common/db.py](common/db.py))
Collection facts and classification labels are split on purpose — do not merge.
- `runs` — run_id, started/finished, config_hash, detection_prompt_hash, ...
- `sites` — crawl queue (see locked decisions). `included`-gated.
- `observations` — one row per (run, site, checkpoint). Paths, nav_method,
  nav_steps, failure_reason. **NO age-gate label.**
- `detections` — filled by 03. One row per (run, site, checkpoint, **model**) →
  enables the 8b/30b/32b comparison with zero re-crawling. `age_gate_type` is
  NULL until manual; upserts never overwrite it.
- `block_checks` — filled by 04, separate so a block page never pollutes a gate.
- `site_outcomes` — one row per (run, site).
All writes are idempotent upserts on natural PKs → restart-safe, no dup rows.

## Failure classification
`blocked_bot` (403/429, Cloudflare/PerimeterX interstitial, CAPTCHA, TLS reset),
`nav_stuck` (step cap / no-progress), `no_such_stage` (site genuinely lacks the
stage — NOT a failure), `timeout`/`error`. Block-page detection is its own pass
(04) so it never contaminates the age-gate label.

## Repo layout
`common/` shared lib (imported, never run) · `config/` global + per-industry YAML
· `scripts/` 01–05 standalone resumable steps · `spike/` Day-1 validation ·
`label_ui/` manual review · `data/` seed (tracked) + screenshots/dom/logs/sqlite
(gitignored). Seed: `data/seed/nicotine-industry-sites.csv` — single column, NO
header, CRLF, ~115k https / 19k http, 0 exact-dup lines, 133,981 rows.

## Environment
- Mac M5 Pro, 48GB, macOS. Python 3.12 in `./.venv` (repo root).
- Ollama running with **all three models pulled**: `qwen3-vl:8b` (6.1 GB, dev),
  `qwen3-vl:30b` (19.6 GB), `qwen3-vl:32b` (20.9 GB) — all Q4_K_M, vision-capable.
- Playwright Chromium not yet installed (`playwright install chromium`, Phase 2).

## Ollama / Qwen3-VL gotchas (verified)
- **Disable thinking** (`think=False`) on every call, else output is polluted
  with chain-of-thought. Detection then returns a bare `1`/`0` in `response`.
- **`format=`-constrained generations route the JSON into the `thinking` field
  and leave `response` empty** on this build (and it's the *fast* path, ~32 eval
  tokens). `vlm._resp_text()` prefers `response`, falls back to `thinking` —
  do not "simplify" that away. Navigation uses `format=NAV_FORMAT`.
- Detection prompt hash (PROVISIONAL — spike phase, prompt still being iterated;
  edit freely, no rehash/lineage bookkeeping until it freezes at first real
  scrape): `6c1d2445c940a95cfb7cdccb53160f28ae57690f89854eda38df7a0c61ceb8c8`.
- Validated live: blank→0, synthetic gate→1; nav click on a YES button rescaled
  to within ~2px of its true center via `coords`.

## How to run (so far)
- Tests: `./.venv/bin/python -m pytest -q`
- Init DB: `./.venv/bin/python -c "from common import db; c=db.connect(); db.init_db(c)"`
- Grounding spike: `./.venv/bin/python spike/spike_grounding.py` (`--mode som` for Set-of-Mark)
- Detection spike: `./.venv/bin/python spike/spike_detection.py`

## Build status (pause for OK between phases)
- [x] **Phase 1 — Scaffold:** tree, requirements, .gitignore, coords + test,
  config loader + YAMLs, SQLite schema, this file. ← awaiting review
- [x] **Phase 2 — spikes + `vlm.py`/`screenshot.py`/`stealth.py` + frozen
  `prompts.py`.** Compiles; VLM path validated live. ← awaiting your spike run
- [ ] Phase 3 — `02_collect.py` (nav loop, resumability, failure class, logging)
- [ ] Phase 4 — `03_classify.py` + `04_classify_blocks.py`
- [ ] Phase 5 — `05_analyze.py` + `label_ui/`
- `01_build_queue.py` (ingest seed → `sites`, offline-tldextract cctld) is a
  **standalone step run before `02_collect`** — not folded into a phase.
