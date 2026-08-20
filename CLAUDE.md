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
6. **The detection prompt AND the relevance prompt are frozen instruments.**
   Stored in ONE place (`common/prompts.py`), hashed, hash recorded per row
   downstream (`runs.detection_prompt_hash` / `sites.relevance_prompt_hash`).
   Never edited inline. `RELEVANCE_PROMPT` determines the **sampling frame**
   itself — at least as consequential for external validity as the detection
   prompt — and gets the same treatment, not a lighter one.

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
- **Frame construction is a 3-script pipeline, in this order:**
  `01_build_queue.py` (ingest ALL seed URLs, normalize + dedupe via
  `common/urls.py`, assign `sample_stratum` immediately — it's a pure
  function of `cctld` via `common/sampling.py:stratum_for_cctld`, no reason
  to defer it) → `01b_screen_relevance.py` (fetch is plain HTTP GET — no
  browser, no screenshots, still ~10x cheaper than a Playwright+VLM screen —
  but the JUDGMENT is an **LLM call**, `common.vlm.classify_relevance()`
  against the frozen `RELEVANCE_PROMPT`, model = `qwen3.6:27b`. NOT a
  keyword/pattern classifier — a deterministic term list structurally can't
  tell a perfume site's "Tobacco Vanille" fragrance note from an actual
  tobacco retailer, which is exactly the false-positive class that motivated
  switching. Still cheap enough to screen the WHOLE frame instead of a
  sample; a block/bot-check page still short-circuits to `"error"`
  deterministically with no LLM call, since recognizing a Cloudflare
  interstitial is mechanical, not a judgment call) → `01c_draw_sample.py`
  (stratified probability sample drawn from `included = 1` sites only —
  **sampling happens AFTER relevance screening**, so the funnel-crawl budget
  is spent on confirmed retailers, not raw CrUX noise). Crawler
  (`02_collect.py`) reads only `included = 1 AND sample_wave IS NOT NULL`.
  Note: `01b` writing `relevance_label` does **not** relax invariant 5 —
  that invariant is about *age-gate* labels and `02_collect.py` specifically;
  relevance is a frame-construction concern, in its own table-adjacent
  columns, produced before any crawling starts. `common/relevance.py` still
  contains the ORIGINAL deterministic `classify_html()` — kept for
  reference/fast local iteration, not called by the production script.
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

## Data model (SQLite, [common/db.py](common/db.py), SCHEMA_VERSION 4)
Collection facts and classification labels are split on purpose — do not merge.
- `runs` — run_id, started/finished, config_hash, detection_prompt_hash, ...
- `sites` — crawl queue (see locked decisions). `included`-gated. v2 added
  `relevance_score`/`relevance_ts`/`http_status` (written by
  `01b_screen_relevance.py` via `db.set_site_relevance()`) and
  `sample_u`/`sample_wave` (written by `01c_draw_sample.py`, write-once via
  `db.set_site_sample_u()`/`db.mark_sites_drawn()`). v3 replaced
  `relevance_ruleset` (INTEGER — a deterministic-classifier concept) with
  `relevance_model`/`relevance_prompt_hash` TEXT, mirroring `detections`'
  `model`/`detection_prompt_hash` provenance now that 01b's judgment is an
  LLM call, not a keyword ruleset. v4 added `relevance_fetch_mode`
  (`'http'` | `'browser'`) recording whether a label came from the plain HTTP
  GET or from the JS-render browser fallback, so the ~2% of sites that are
  invisible to a plain fetch are identifiable in the data rather than
  silently indistinguishable. `upsert_site()` is the FRAME writer only
  (site_id, host, cctld, ...) — it never touches these relevance/sampling
  columns directly, so a re-run of `01_build_queue` structurally cannot wipe
  a relevance screen or a sample draw. `init_db()` raises
  `SchemaVersionMismatch` rather than silently re-stamping an old DB — a
  version bump means new columns the code depends on.
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
`label_ui/` manual review · `data/` seed (tracked) + screenshots/dom/logs/html/sqlite
(gitignored, except seed). `data/html/<site_id>.html.gz` is `01b_screen_relevance.py`'s
stored fetch — the source of truth for that step, so `--reclassify` re-scores
it with zero network calls. Seed: `data/seed/nicotine-industry-sites.csv` —
single column, NO header, CRLF, 114,714 https / 19,268 http, 0 exact-dup lines,
**133,982 rows** (file has no trailing newline — a naive `split('\n')` silently
drops the last row; stream with `for line in f` instead). After www/port
canonicalization the distinct-site frame is **115,792 rows** (13.6% collapse) —
verified by `01_build_queue.py`, which also assigns each row to one of 9
ccTLD-derived `sample_stratum` buckets (`common/sampling.py`): gtld 65.5%,
eu_east 12.7%, eu_west 7.9%, apac 4.8%, latam 3.2%, uk_ie 2.5%, mena_af 1.4%,
other 1.0%, na 0.9%.

## Environment
- Mac M5 Pro, 48GB, macOS. Python 3.12 in `./.venv` (repo root).
- Ollama running with **four models pulled**: `qwen3-vl:8b` (6.1 GB, dev),
  `qwen3-vl:30b` (19.6 GB), `qwen3-vl:32b` (20.9 GB) — Q4_K_M, vision-capable —
  plus `qwen3.6:27b` (17.4 GB), a dedicated TEXT model (no vision) used only
  by `classify_relevance()` for frame-relevance screening.
- Playwright Chromium installed.

## Ollama / Qwen3-VL gotchas (verified)
- **Disable thinking** (`think=False`) on every call, else output is polluted
  with chain-of-thought. Detection then returns a bare `1`/`0` in `response`.
- **`format=`-constrained generations route the JSON into the `thinking` field
  and leave `response` empty** on this build (and it's the *fast* path, ~32 eval
  tokens). `vlm._resp_text()` prefers `response`, falls back to `thinking` —
  do not "simplify" that away. Navigation uses `format=NAV_FORMAT`.
- **Detection prompt hash — FROZEN** (2026-06-28, model = `qwen3-vl:30b`):
  `eab9754285c7e790cfbd97aa3c8b916434a4feaf31df7f7c7ea6896039ae987a`.
  Validated on `spike/spike_detection.py`'s n=23 hand-verified set: 23/23
  correct (TP=9 FP=0 TN=14 FN=0) on both full_page and viewport captures.
  Specifically tests the language-agnostic clause: 6 non-English age gates
  (bg/pt-BR/pl/fr/nl/it), all correctly detected — including two where the
  real gate was partially obscured by a stacked cookie-consent banner, and
  the model correctly saw past the banner without miscounting it as a gate.
  Also validates the exclusion boundary: bot "not a robot" checks, Cloudflare
  block/verification pages, and a maintenance page were all correctly NOT
  flagged. Per invariant #6, any further wording change is a new instrument —
  edit `DETECTION_PROMPT` deliberately and treat results under the new hash
  as a separate measure.
- Validated live: blank→0, synthetic gate→1; nav click on a YES button rescaled
  to within ~2px of its true center via `coords`.
- **Relevance prompt hash — FROZEN** (model = `qwen3.6:27b`):
  `ad59a3d194fe2deec67c68d043340a1f7bc5bf4c5886361822caa7d93113a2d7`.
  (Superseded `ffd9b7b7…` after the maintenance-rule fix below; nothing had
  been screened under the old hash, so no data was invalidated.)
  Validated on `spike/spike_relevance.py`'s hand-verified set: 13/13 correct
  on genuine judgment calls (2 apparent mismatches were non-classifier
  artifacts — a JS-rendered storefront whose real content never reaches a
  no-browser fetch, and a same-session repeated-request rate limit — not
  prompt/model errors). Two real fixes came out of live testing before
  freezing: (1) an explicit instruction that a nicotine-adjacent WORD in a
  product/brand name (e.g. a perfume's "Tobacco Vanille" fragrance note) is
  not evidence of a nicotine retailer — judge actual business, not word
  matches; (2) an explicit instruction that "under construction"/"coming
  soon"/parked-domain placeholders classify as `unrelated` regardless of
  what the domain name implies, since the model was initially over-weighting
  the domain/title over the page's actual (empty) content. A third fix
  followed the n=150 gold-set scoring: rule (2) was too blunt and sent a
  real vape shop showing a maintenance banner *over its full catalogue nav*
  (vapebuddy.ch) to `unrelated`, so the rule now distinguishes a
  placeholder-ONLY page from a working shop that is briefly down.
- **Relevance scored against the hand-labeled gold set** (n=150,
  `data/labels/relevance_goldset_claude-handlabeled_n150.txt`).
  Before the maintenance-rule fix and browser fallback: P .979 / R .922 /
  F1 .949. **After both: precision .980 / recall .980 / F1 .980**
  (TP=50 FN=1 FP=1 TN=98). Three rows flipped, all correct, none regressed:
  `musthavetobacco.ru` (via browser render), `vapebuddy.ch` (maintenance
  rule), `rainbowvapes.co.uk` (a transient fetch failure that recovered).
  Recall is flat across languages — 28/30 non-English-market ccTLDs vs 19/21
  English — the multilingual bias risk retired empirically.
  Browser fallback fired on 7/150 sites (4.7%) and was correct on all 7, so
  it recovers the blind spot without costing precision.
  Two residual errors, both defensible: `bookmysmoke.com` (a BORDERLINE
  hookah-RENTAL service — arguably a gold-label judgment call, not a model
  error) and `legal.blinkersvape.co` (an Indonesian gambling portal squatting
  a vape domain with vape-flavoured marketing copy — deceptive by design).
  Throughput 5.6s/site end-to-end (median LLM latency 3.2s) → full-frame
  screen ≈ 7.5 days serial.
- `classify_relevance()` is a TEXT-only call — no `images=` kwarg, unlike
  `detect_age_gate()`/`decide_navigation()`. `_parse_relevance()` and
  `_parse_nav()` both had a latent crash bug: `json.loads()` succeeds on any
  valid JSON scalar (`null`, a bare string/number, a list), and calling
  `.get()` on a non-dict result raised an unhandled `AttributeError` instead
  of degrading gracefully. Both now check `isinstance(obj, dict)` before
  calling `.get()` — a single malformed model response must never crash a
  multi-day screen or a live navigation loop.

## Navigation findings (spike/spike_navigation.py — READ BEFORE BUILDING 02_collect)
First code that actually executes actions; `spike_grounding.py` only ever drew
a crosshair. Run over 8 gold-set retailers, walking
`site_arrival → store → product → add_to_cart` (deliberately stops before the
terminal checkpoint). Depth reached: **site_arrival 8/8, store 5/8, product
4/8, add_to_cart 2/8.**

1. **Qwen3-VL often returns a 2-element POINT `[x,y]` instead of the 4-element
   box the prompt asks for.** `_parse_nav` required exactly 4 and silently
   downgraded those to `stuck` — on the first site tested, the model had
   correctly located the age-gate button and the click was thrown away.
   `vlm._normalize_box()` now widens a point to `[x,y,x,y]`, whose centre is
   the point, so `coords.box_center_to_pixel()` is unchanged. Regression tests
   in `tests/test_vlm.py`.
2. **Age gates block the crawl** — the thing being measured is also an
   obstacle. A gate modal swallows clicks aimed at the nav behind it. The
   navigation prompt (NOT a frozen instrument) now instructs: dismiss blocking
   overlays first via the AFFIRMATIVE control, never the "No/exit" one, in any
   language, and never attempt a CAPTCHA (invariant #2). This is
   methodologically clean — the checkpoint screenshot captures the gate before
   it is dismissed, and detection is a downstream pass (invariant #5).
3. **The model would not stop when the goal was met** — it reached the store
   listing, then filtered, wandered to the delivery page and back. That is a
   DATA-QUALITY bug, not just wasted time: the page on screen when the loop
   ends is the page recorded for that checkpoint. The prompt now makes
   "is the goal ALREADY satisfied? → done" the explicit first step.
4. **`loop.step_cap: 6` is NOT the binding constraint.** Every one of the 11
   successful checkpoints converged in ≤3 steps. All three `step_cap`
   failures had progressed on 100% of their clicks — they were looping
   (catalog → filter → catalog → delivery → catalog), not running out of
   budget. Raising the cap would only buy more wandering.
   **Design implication for 02_collect:** `no_progress_limit` only catches
   *unchanged* URL+DOM, so it cannot see this loop. Add a **URL-revisit
   detector** — returning to a URL already seen within the same checkpoint is
   the real stuck signal here.
5. Executed actions changed the page 69% of the time (38/44). Median nav
   latency 2.7s; per-site funnel walk ≈35s for this 4-checkpoint prefix.

## How to run (so far)
- Tests: `./.venv/bin/python -m pytest -q`
- Init DB: `./.venv/bin/python -c "from common import db; c=db.connect(); db.init_db(c)"`
- Grounding spike: `./.venv/bin/python spike/spike_grounding.py` (`--mode som` for Set-of-Mark)
- Detection spike: `./.venv/bin/python spike/spike_detection.py`
- Relevance spike: `./.venv/bin/python spike/spike_relevance.py`
- Navigation spike: `./.venv/bin/python spike/spike_navigation.py` (`--headful` to watch)
- Build queue: `./.venv/bin/python scripts/01_build_queue.py`
- Screen relevance: `./.venv/bin/python scripts/01b_screen_relevance.py --limit N`
- Draw sample: `./.venv/bin/python scripts/01c_draw_sample.py --dry-run` (review before a real draw)

## Build status (pause for OK between phases)
- [x] **Phase 1 — Scaffold:** tree, requirements, .gitignore, coords + test,
  config loader + YAMLs, SQLite schema, this file.
- [x] **Phase 2 — spikes + `vlm.py`/`screenshot.py`/`stealth.py` + frozen
  `prompts.py`.** Detection prompt FROZEN (n=23, 6 languages, see gotchas below).
- [x] **Frame construction (standalone, run before Phase 3 — not itself a
  phase):** `01_build_queue.py` → `01b_screen_relevance.py` → `01c_draw_sample.py`.
  All three built, tested (`common/urls.py`, `common/sampling.py`,
  `common/relevance.py`, schema v3), and verified live end-to-end against the
  real seed (frame = 115,792 sites, 0 rejects, idempotency proven against a
  live DB, `--reclassify` and `--retry-failed` both verified). Models switched
  to `qwen3-vl:30b` for both navigation and detection. `01b` initially used a
  deterministic keyword classifier (schema v2); switched to an LLM judgment
  (`qwen3.6:27b`, frozen `RELEVANCE_PROMPT`, schema v3) after live testing
  found real false positives/negatives a keyword list can't fix — see the
  Ollama gotchas below. Real observed throughput: ~3.2s/site (LLM-call-bound,
  not fetch-bound), so a full-frame screen is now estimated at ~4-5 days
  serial rather than 1-2 — this has NOT yet been run at full scale, only in
  small live batches (largest: 40 sites).
- [ ] Phase 3 — `02_collect.py` (nav loop, resumability, failure class, logging)
- [ ] Phase 4 — `03_classify.py` + `04_classify_blocks.py`
- [ ] Phase 5 — `05_analyze.py` + `label_ui/`
