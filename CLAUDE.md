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
4a. **Chunks are a scheduling unit, not a work-distribution unit.**
   `chunk_runs` (schema v5, see below) has no lease, claim, owner, or
   heartbeat column, and never will — two processes must never run two
   chunks of the same set concurrently. A stale `running` row (left by
   `kill -9`) is the crash signature, not a lock: the next invocation simply
   overwrites it, printing a loud warning but never blocking. For the same
   reason, do not add retry-on-`SQLITE_BUSY` wrappers.
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
  Also the screenshot/dom folder name — **AMENDED for collection**, see below.
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
- **`sites.industry` (a DB value) and `config/<name>.yaml`'s filename are TWO
  DIFFERENT NAMESPACES** that happen to collide for alcohol/gambling but not
  nicotine/vape: the default seed yields `sites.industry="nicotine"`, but
  checkpoints live in `config/vape.yaml` (`industry: "vape"`). Bridged by
  `common.config.config_name_for_sites_industry()` — identity for anything
  not explicitly remapped. Every checkpoint-reading script must go through
  this, not hardcode `"vape"` the way `spike_navigation.py` does.
- **AMENDED (2026-08-23):** canonical collection screenshots additionally
  nest under `run_id`: `data/screenshots/<run_id>/<site_id>/<checkpoint>_{fullpage,viewport}.png`
  (and `dom_path`, if ever populated) — **not** `data/screenshots/<site_id>/...`
  as originally stated above for `site_id` alone. Added when
  `02_collect_browseruse.py` was built: a `--retry-blocked` re-attempt under
  a NEW `run_id` would otherwise silently overwrite the first attempt's
  images on disk while that attempt's `observations` row still points at
  the (now wrong) path. `data/html/<site_id>.html.gz` (01b's fetch store) is
  unaffected — that's inherently a one-per-site, no-run-concept artifact.

## Data model (SQLite, [common/db.py](common/db.py), SCHEMA_VERSION 5)
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
  version bump means new columns the code depends on. v5 added
  `sites.relevance_attempts` (incremented by every `set_site_relevance()`
  call), backing a real `relevance.max_attempts` config cap (default 3) —
  previously `iter_pending_relevance`'s docstring *promised* this cap but
  nothing enforced it, so unbounded `--retry-failed` could hammer a
  permanently-dead host forever and a chunk containing it could never reach
  `done`.
- **`chunk_sets` / `chunk_members` / `chunk_runs`** (v5) — addressable,
  resumable work chunks shared by `01b_screen_relevance.py` today and
  (later) `02_collect.py`/`03_classify.py`. Partition math lives in
  [common/chunking.py](common/chunking.py) (pure, DB-free, mirrors
  `common/sampling.py`'s role); the CLI is `scripts/chunk_work.py`
  (`create`/`extend`/`verify`/`status`/`unstamp`), deliberately unnumbered —
  a cross-cutting utility, not a pipeline stage.
  **Rule of one truth:** a site's completion state is a property of the
  site's OWN row (`sites.relevance_prompt_hash` / `site_outcomes` /
  `detections`). `chunk_members` decides which sites an invocation may
  *touch* (write-once, trigger-enforced — `common.chunking.partition()`'s
  output, keyed by `sha256(chunk_seed|generation|site_id)`, deliberately
  **not** `sampling.frame_salt` — reusing that salt would make chunk 0
  exactly the sites `01c` draws first). `chunk_runs` records *what
  happened*, one row per `(chunk_set_id, chunk_index, consumer, scope)`. **No
  script ever reads `chunk_runs` to decide whether to do work** — work
  selection is always the indexed join `pending ⋈ chunk_members`
  (`db._pending_relevance_sql()`, feeding both `iter_pending_relevance()` and
  `count_pending_relevance()` so the queue has exactly one definition), and
  `--chunk next` resolves via `db.next_pending_chunk_index()` from that same
  join, never from `chunk_runs.status`. `scope` namespaces "done" per
  consumer (`RELEVANCE_PROMPT_HASH` for 01b; will be `run_id` for 02;
  `"run_id|model"` for 03) — a frozen-prompt revision reconciles for free:
  the old scope's `chunk_runs` rows stay `done` forever (a true historical
  statement about the old instrument), and under the new hash the same
  partition simply reports pending again with no cleanup. A chunk's `status`
  is always computed from the **base** queue definition (prompt hash only —
  no `--retry-failed`/`--industry` widening), so a wide or narrow invocation
  can never regress an already-`done` chunk. Within a chunk,
  `01b_screen_relevance.py` iterates by `member_rank` and bypasses
  `_interleave_by_stratum` — a uniformly random partition makes any prefix of
  any chunk an unbiased subsample of the frame, which round-robin
  interleaving doesn't buy you and would only complicate. Per-chunk JSONL logs
  land in `data/logs/chunks/<set>/c<idx:05d>__<consumer>__<scope[:12]>__<ts>.jsonl`,
  one file per attempt (meta / per-site / summary lines) — see "Repo layout"
  below for why this needs a backup story.
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
· `scripts/` numbered pipeline stages (01–05) **plus two cross-cutting/
parallel scripts that don't fit the numbering:** `chunk_work.py`
(deliberately unnumbered — chunking is a scheduling concern used BY
multiple stages, not itself a stage) and `02_collect_browseruse.py` (a
SECOND navigation engine parallel to the still-unbuilt numbered
`02_collect.py`, not a replacement for it — see Build status) · `spike/`
Day-1 validation · `label_ui/` manual review · `data/` seed (tracked) +
screenshots/dom/logs/html/sqlite (gitignored, except seed). `data/html/<site_id>.html.gz`
is `01b_screen_relevance.py`'s stored fetch — the source of truth for that
step, so `--reclassify` re-scores it with zero network calls. Seed:
`data/seed/nicotine-industry-sites.csv` —
single column, NO header, CRLF, 114,714 https / 19,268 http, 0 exact-dup lines,
**133,982 rows** (file has no trailing newline — a naive `split('\n')` silently
drops the last row; stream with `for line in f` instead). After www/port
canonicalization the distinct-site frame is **115,792 rows** (13.6% collapse) —
verified by `01_build_queue.py`, which also assigns each row to one of 9
ccTLD-derived `sample_stratum` buckets (`common/sampling.py`): gtld 65.5%,
eu_east 12.7%, eu_west 7.9%, apac 4.8%, latam 3.2%, uk_ie 2.5%, mena_af 1.4%,
other 1.0%, na 0.9%.

**`data/logs/chunks/` is gitignored and is the ONLY record of a multi-day
run's per-site timing and provenance** (meta/site/summary JSONL, one file per
chunk attempt — see the chunking entry in "Data model" above). Unlike
screenshots or the sqlite DB, this is not regenerable from anything else once
a chunk's real-world attempt is over — it needs a backup story (e.g. a
periodic `rsync`/archive off-machine) before the full 47-chunk frame screen
runs unattended for days. Not yet set up as of this writing.

## Environment
- Mac M5 Pro, 48GB, macOS. Python 3.12 in `./.venv` (repo root).
- Ollama running with **four models pulled**: `qwen3-vl:8b` (6.1 GB, dev),
  `qwen3-vl:30b` (19.6 GB), `qwen3-vl:32b` (20.9 GB) — Q4_K_M, vision-capable —
  plus `qwen3.6:27b` (17.4 GB), a dedicated TEXT model (no vision) used only
  by `classify_relevance()` for frame-relevance screening.
- Playwright Chromium installed.
- A SECOND, separate local inference backend for `02_collect_browseruse.py`
  only: `mlx`/`mlx-lm` 0.31.x already installed in `~/.pyenv/versions/3.12.3`
  (not this project's `.venv` — MLX is not a Python dependency of this
  repo, it runs as an independent local server process, like Ollama).
  `mlx-vlm` (the vision-serving package `bu-30b-a3b-preview` actually needs)
  is NOT installed anywhere yet.
- **Memory contention across backends:** `qwen3.6:27b` alone is 17.4 GB;
  running Ollama's relevance/detection models AND a new MLX server holding
  `bu-30b-a3b-preview` (likely a similar order of magnitude, given it's also
  a ~30B model in 4-bit) at the same time, on a 48GB machine, alongside
  Chromium, is tight. Sequence a relevance screen and a browser-use
  collection run rather than overlapping them; don't treat timing measured
  under both running at once as reliable.

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
- **Gold set extended to n=190** (2026-08-20, `spike/score_relevance_goldset.py`
  — a reproducible replacement for the original ad-hoc scratchpad scoring
  run). 40 new rows hand-labeled from already-fetched candidate pages (no new
  crawling); **only 15/40 (37.5%) turned out relevant**, not the ~45%
  expected going in from a keyword pre-filter — mostly "juice"/"liquidation"
  word collisions, domain-parking pages, and Cloudflare interstitials the
  keyword hint couldn't distinguish from real retailers, itself a small
  live demonstration of exactly the false-positive class that motivated an
  LLM judgment over a keyword classifier in the first place. Overall balance
  barely moved: 34.0% → 34.7% positive (66/190).
  Full n=190, scored live (browser fallback on, matching production):
  **precision .969 / recall .955 / F1 .962** (TP=63 FP=2 TN=122 FN=3).
  The new 40 scored **40/40 (100%)** — every hand-label agreed with the live
  model, a good calibration signal for the new labels. The ORIGINAL 150,
  re-scored the same run, came back **precision .960 / recall .941 / F1 .950**
  — lower than the frozen .980/.980/.980, but not a prompt/model regression:
  of the 5 total mismatches (all 5 fell in the original 150, none in the new
  40), 2 are the SAME residual errors already on record above
  (`bookmysmoke.com`, `legal.blinkersvape.co`), 2 are sites that went fully
  unreachable/started blocking sometime in the ~2 months since the June
  freeze (`gute-liquids.de`, and `rainbowvapes.co.uk` — already flagged
  above as a site prone to transient fetch failure), and only 1
  (`blizyn.lento.pl`, PL classifieds mistaken for `marketplace`) is a
  genuinely new near-miss. **Caveat for future re-scoring:** unlike
  `01b_screen_relevance.py`'s production path, which stores fetched HTML
  specifically so `--reclassify` can re-score with zero network calls, this
  scorer always re-fetches live — so a re-run's numbers conflate "did the
  prompt/model change" with "did the live web change," which is why the
  original-150 subset didn't reproduce exactly. A future revision could
  cache goldset fetches the same way if exact reproducibility across time
  becomes important.
- `classify_relevance()` is a TEXT-only call — no `images=` kwarg, unlike
  `detect_age_gate()`/`decide_navigation()`. `_parse_relevance()` and
  `_parse_nav()` both had a latent crash bug: `json.loads()` succeeds on any
  valid JSON scalar (`null`, a bare string/number, a list), and calling
  `.get()` on a non-dict result raised an unhandled `AttributeError` instead
  of degrading gracefully. Both now check `isinstance(obj, dict)` before
  calling `.get()` — a single malformed model response must never crash a
  multi-day screen or a live navigation loop.

## MLX / browser-use gotchas (verified, 2026-08-23)
`scripts/02_collect_browseruse.py` is a second, parallel navigation engine
(build-order steps 2-3 of 5 done — single hardcoded site (`--test-site-id`)
verified live end-to-end against `vape.bg`, 10 live-test iterations) built
on the `browser-use` library's `Agent`, driven by a Qwen3-VL-30B-A3B-Instruct
fine-tune (`bu-30b-a3b-preview`) not available on Ollama. `qwen3.6:27b` and
`qwen3-vl:30b` are UNCHANGED and stay on Ollama — see `config/global.yaml`'s
`navigation_bu` block.

- **`browser-use` unconditionally installs `openai`, `anthropic`,
  `google-genai` (+ the whole google-api/-auth stack), and `groq` as CORE
  dependencies, at every released version back to 0.1.0** — confirmed via
  PyPI metadata across versions, not a recent change. It supports many LLM
  backends and ships all their client SDKs regardless of which one you
  actually use; there is no lighter extras group that excludes them. This
  repo only ever constructs
  `browser_use.ChatOpenAI(base_url="http://localhost:<port>/v1", api_key="not-needed")`
  — no client from any other bundled SDK is ever instantiated. `requirements.txt`'s
  "no hosted-LLM SDK" comment was amended to be about network destination,
  not package presence, given this.
- **`browser-use` phones home via PostHog telemetry by default** (`posthog`
  is itself a core dependency). `ANONYMIZED_TELEMETRY=false` MUST be set
  before the first `import browser_use` anywhere — `scripts/02_collect_browseruse.py`
  sets it via `os.environ.setdefault(...)` at the top of the file, before
  its own `from browser_use import ...` line.
- **`max_steps` is NOT an `Agent()` constructor argument in 0.13.8** —
  confirmed via `inspect.signature`. It belongs to `agent.run(max_steps=N)`.
  `Agent.__init__` has a catch-all `**kwargs`, so passing `max_steps=6` to
  the constructor would NOT raise — it would silently vanish, and the agent
  would run to browser-use's actual default of 500 steps per checkpoint
  instead of the intended cap. A real bug in an early draft of this
  script's design; caught by inspecting the installed package directly
  rather than trusting a supplied code snippet.
- **`BrowserSession`/`BrowserProfile` expose ZERO fields for locale,
  timezone, or fingerprint-hardening — there is no "stealth" concept in
  this library at all**, and `BrowserSession.get_current_page()` returns
  browser-use's OWN internal CDP-based `actor.page.Page`, not
  `playwright.sync_api.Page` — confirmed `common/stealth.py`/
  `common/screenshot.py` cannot be called on it. **The CDP-bridge approach
  (a separate Playwright client connected via `BrowserSession.cdp_url`) was
  ABANDONED after live testing**, not just found imperfect — see the
  `keep_alive` entry below for why. **What actually shipped:**
  - Canonical screenshots: `BrowserSession.take_screenshot(path=..., full_page=...)`
    — browser-use's own native primitive (the same one its debug-trail
    screenshots already used reliably), not `common/screenshot.py`. That
    module's Playwright-`page.screenshot()`-based functions are confirmed
    NOT reusable for this engine at all, not just "need adaptation."
  - Stealth: best-effort injection of `playwright_stealth.Stealth().script_payload`
    (a plain JS string) via `actor_page.evaluate(script)` — no second
    connection needed. Real, documented limitation: this runs AFTER the
    page has already loaded, not as an init-script before any page JS
    executes, so it cannot undo whatever the page's own scripts already
    read/set on first load. Not equivalent to real init-script stealth —
    a residual gap, not silently claimed as fixed.
  - Locale/timezone: no native `locale=`/`timezone_id=` exists either — set
    via raw Chromium launch args (`args=["--lang=..."]`) and process env
    (`env={"TZ": ...}`) instead, translating
    `common.stealth.profile_for_cctld()`'s output to the lower-level
    equivalent.
- **ROOT CAUSE of 7 straight failed live-test iterations, found by reading
  `browser_use/agent/service.py` directly: `Agent.close()` — called
  automatically at the end of EVERY `agent.run()` — checks
  `browser_session.browser_profile.keep_alive`. If falsy (the default, and
  what an early draft of this script used), it calls `session.kill()`, a
  FULL hard teardown of the shared browser session, unconditionally, after
  every single checkpoint.** This is what made a CDP-bridge connection go
  stale/`ECONNREFUSED`, and even made browser-use's OWN
  `take_screenshot()` raise `AssertionError: Root CDP client not
  initialized` after checkpoint 1 — it was never a capture-strategy bug,
  the shared session was being destroyed out from under everything. **Fix:
  `BrowserProfile(keep_alive=True, ...)`.** This is the literal precondition
  for the original spec's core architecture (one `BrowserSession` reused
  across checkpoint sub-agents, carrying cart/cookie state) to hold at all
  — without it, every checkpoint silently starts a brand-new browser.
  Correspondingly, final cleanup must call `session.kill()`, not
  `session.stop()` — with `keep_alive=True`, `.stop()` alone "keeps the
  browser alive" per its own docstring, which would leak a Chromium
  process per site once a real site loop exists.
- **`use_thinking=True` (Agent's default) caused every step to time out at
  60s** against `bu-30b-a3b-preview` — confirmed NOT an inference-speed
  problem (an isolated `ChatOpenAI.ainvoke()` call with a real page
  screenshot completed in 4.3s). Fixed with `use_thinking=False`.
  Residual, unresolved finding: even with thinking disabled, this model's
  regular structured output (memory/eval/plan reasoning embedded IN the
  JSON action fields, not gated by `use_thinking`) varies wildly in length
  step to step — some calls ~4s, others exceeded even `max_completion_tokens=16384`
  (raised twice, from 4096 → 8192 → 16384, the judge call and even
  occasional regular action steps still truncated at the highest setting
  tried). `step_timeout`/`llm_timeout` raised 60→180s. Neither value is
  confirmed sufficient for every case — a real, open operational
  characteristic of this fine-tune on this hardware, not fully tuned.
- **Classification bug found live: `is_done()` alone is not enough.**
  It only reports "did the agent call `done()` at all" (vs. exhausting
  `max_steps`) — nothing about whether the agent's OWN `done(success=...)`
  call reported success or failure. Confirmed live: an agent gave up on the
  `cart` checkpoint after repeatedly failing to click a cart icon,
  explicitly called `done(success=False)` admitting it never left the
  product page, and a check of `is_done()` alone let this through as
  `reached=1`/`"completed"` — the checkpoint's own recorded step URLs
  showed all 6 attempts stuck on the identical product-page URL. Fixed:
  `history.is_done() and history.is_successful()`.
  `is_successful()` reads `last_result.success` — the agent's own
  `done()` parameter, from `browser_use/agent/views.py` — NOT the separate,
  frequently-truncating judge mechanism (`use_judge=True`'s secondary LLM
  call, which failed/truncated in most live tests here); it is guaranteed
  non-None whenever `is_done()` is True, so the combined check is safe.
- **`enable_default_extensions` defaults to True**, silently injecting an
  ad-blocker (uBlock Origin), an auto-cookie-consent-dismisser ("I still
  don't care about cookies"), and a URL cleaner (ClearURLs) into every
  session — confirmed live via the first test run's log (all three
  downloaded and loaded, unasked). **Disabled explicitly**
  (`enable_default_extensions=False`): a real methodological risk for this
  project specifically, not a generic nuisance — invariant #5 requires
  screenshots to be the unmodified source of truth, and some sites
  implement age gates with the same modal/banner pattern as cookie-consent
  banners. An extension silently auto-dismissing something before the
  canonical screenshot is an invisible confound: the image would simply
  show no gate, indistinguishable from a site that never had one.
- **A `captcha_solver` field defaults to `True`** (not False) on
  `BrowserProfile` — CAPTCHA-solving capability is ON by default in this
  library. Per invariant #2, explicitly set `captcha_solver=False`; it is
  NOT automatically absent just because nothing asks for it.
- **Confirmed live, end to end (2026-08-23):** the agent correctly located
  and dismissed a real Bulgarian-language age-verification gate on
  `vape.bg` ("ДА, ПРОДЪЛЖИ" — "YES, CONTINUE") across every successful
  test run, then walked `store → product → add_to_cart → cart`, reaching a
  genuine `shopping-cart.html` page (product, quantity, price, and total
  all correctly visible in the stored screenshot, plus an "ORDER" button —
  never clicked, never even part of any constructed task) — a complete,
  safe funnel walk matching invariant #1 exactly.
- **`run_id` has no default in `scripts/02_collect_browseruse.py` and must
  be passed explicitly, reused across every invocation of the same logical
  crawl wave.** Unlike 01b's chunk scope (a stable prompt hash across the
  whole multi-day screen), this script's `chunk_runs` scope IS `run_id` —
  auto-generating a fresh one per invocation (the originally supplied spec's
  default) would silently make `site_outcomes`-based resume meaningless
  (every invocation would see 100% pending regardless of prior progress)
  without ever raising an error. Deliberately made a required, no-default
  argument instead.
- **`sites.industry` (a DB value, e.g. `"nicotine"`) and `config/<name>.yaml`'s
  filename (e.g. `"vape"`) are different namespaces** — see the Locked
  design decisions entry above. `--industry` on this script takes the DB
  value; `common.config.config_name_for_sites_industry()` bridges it.
- `mlx`/`mlx-lm` are already installed (in `~/.pyenv/versions/3.12.3`,
  **not** this project's `.venv` — mlx/mlx-vlm are not a dependency of this
  repo at all, they run as an independent local server process like
  Ollama). `mlx_lm.server` is confirmed live and working but **text-only,
  no vision/image flags whatsoever** — it cannot serve `bu-30b-a3b-preview`.
- **`mlx-vlm` installed and verified live (2026-08-23).** `pip install
  mlx-vlm` in that same pyenv env pulled in `mlx-vlm==0.6.15` and, as a side
  effect, upgraded `mlx` 0.31.2→0.32.1 / `mlx-metal` likewise / `transformers`
  5.9.0→5.15.1 — `mlx_lm` still imports fine afterward, no breakage
  observed, but that shared environment's versions moved. `python -m
  mlx_vlm.server --port 8099` (no `--model` flag) started cleanly in ~4s and
  `/v1/models` correctly listed the pre-existing local HF cache. Reading the
  actual routing code (`mlx_vlm/server/app.py:get_cached_model`) confirmed
  the model is resolved **dynamically per request** from whatever string
  the client sends as `model` — checked as a local directory first, else
  treated as an HF repo ID and downloaded+cached on first use. `--model` at
  startup is only an optional pre-load, never required. `--api-key` is also
  optional — omit it and no auth is enforced, matching
  `ChatOpenAI(api_key="not-needed")`. `mlx_vlm.convert` (a real,
  `--hf-path`/`--mlx-path`/`-q`/`--q-bits` HF→MLX conversion tool) exists
  for the day a target model isn't already in MLX format — not needed for
  `bu-30b-a3b-preview` specifically, see below.
- **`bu-30b-a3b-preview` resolved and verified real via the HF Hub API**:
  `mlx-community/bu-30b-a3b-preview-4bit` — `qwen3_vl_moe` architecture,
  tagged `image-text-to-text` (vision-capable) and `browser_use`, base model
  `browser-use/bu-30b-a3b-preview` (i.e. this is browser-use's own published
  fine-tune, quantized to MLX 4-bit by mlx-community, not a mystery private
  checkpoint). 4 safetensors shards, **~18.3 GB total** — confirmed via the
  Hub API's blob sizes before recommending the download. Already the value
  of `config/global.yaml`'s `navigation_bu.llm_model`.
- The preflight check's CODE was first proven correct by pointing it at
  Ollama's own OpenAI-compatible endpoint as a stand-in (see above), THEN
  confirmed live against the real `mlx_vlm.server` + `bu-30b-a3b-preview`:
  server started with `python -m mlx_vlm.server --port 8080` (no `--model`
  flag — loads dynamically per request, confirmed), preflight replied
  `"Red"` in ~10s cold / ~1s warm, and the full single-site crawl completed
  successfully from there. `mlx_vlm.server`'s `--host`/`--port`/`--api-key`
  flags all matched what was documented, live-verified, not just assumed.

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
- Create a chunk set: `./.venv/bin/python scripts/chunk_work.py create --set frame-v1 --preset frame --chunk-size 2500 --consumer 01b_screen_relevance --dry-run` (drop `--dry-run` to commit)
- Verify a chunk set's reproducibility: `./.venv/bin/python scripts/chunk_work.py verify --set frame-v1`
- Check chunk progress: `./.venv/bin/python scripts/chunk_work.py status --set frame-v1 --failed`
- Screen one chunk: `./.venv/bin/python scripts/01b_screen_relevance.py --chunk-set frame-v1 --chunk 0` (or `--chunk next` to resolve the lowest pending chunk automatically)
- browser-use collection preflight only: `./.venv/bin/python scripts/02_collect_browseruse.py --industry nicotine --dry-run` (requires `config/global.yaml`'s `navigation_bu.llm_model` to be set and a vision-capable OpenAI-compatible server reachable at `llm_base_url` — see the MLX / browser-use gotchas section)

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
  Ollama gotchas below. Measured end-to-end throughput is 5.6s/site (median
  LLM latency 3.2s of that) → full-frame screen ≈ **7.5 days serial** (the
  correct, measured figure — an earlier LLM-latency-only estimate of ~4-5
  days quoted here was stale and has been removed). This has NOT yet been
  run at full scale, only in small live batches (largest: 40 sites).
- [x] **Chunking primitive (schema v5, standalone utility, not itself a
  phase):** `common/chunking.py` (pure partition math) + `scripts/chunk_work.py`
  (`create`/`extend`/`verify`/`status`/`unstamp`) + `01b_screen_relevance.py
  --chunk-set/--chunk`. Makes the 7.5-day frame screen addressable as ~47
  independently resumable 2500-site chunks instead of one all-or-nothing
  run. Built, unit-tested, and **pilot-verified live against the real
  115,792-site frame**: `frame-v1` chunk set created and `verify`d (47
  chunks), `--chunk 0 --limit 20` run and its JSONL + stored HTML inspected
  by hand, resume confirmed, a REAL `SIGINT` (via `subprocess.Popen` +
  `send_signal` — a backgrounded shell `&` silently no-ops on SIGINT in a
  non-interactive shell, a real gotcha hit and fixed mid-session) produced
  an accurate `status=interrupted`, resume-after-interrupt was clean, a
  full chunk reached `done`, its re-run was a loud no-op, and `--chunk next`
  correctly skipped it. The full 47-chunk frame screen itself has NOT been
  launched — that remains a separate go-ahead.
- [ ] Phase 3 — `02_collect.py` (native Qwen3-VL custom-loop nav engine —
  nav loop, resumability, failure class, logging). Unbuilt.
- [~] **`02_collect_browseruse.py`** — a SECOND, parallel navigation engine
  (not a Phase 3 substitute) built on the `browser-use` library's `Agent` +
  a locally-served MLX vision model (`mlx-community/bu-30b-a3b-preview-4bit`),
  specifically so its reach-rate is directly comparable against whatever
  Phase 3 eventually produces over the same drawn sample. Build-order steps
  2-3 of 5 done. Step 3 (single hardcoded URL, `--test-site-id`, all 5
  checkpoints, both captures, debug trail, DB writes) is **verified live and
  working end-to-end** against `vape.bg` after 10 live-test iterations —
  see the MLX / browser-use gotchas section above for the concrete bugs
  found and fixed along the way (`keep_alive`, `use_thinking`,
  `enable_default_extensions`, the `is_done()`-vs-`is_successful()`
  classification bug). Confirmed by hand: the agent correctly finds and
  dismisses a real Bulgarian age gate, walks `site_arrival → store →
  product → add_to_cart → cart`, and STOPS at a genuine cart page (product,
  price, total, and an "ORDER" button all visible in the stored screenshot)
  without ever approaching it — `checkout_initiation` is never loaded, no
  purchase task is ever constructed, matching invariant #1. Steps 4-5 (the
  chunk-aware site loop consuming `chunk_work.py --preset drawn-sample`,
  `--retry-blocked`, restart cadence, kill/resume verification) are NOT
  started — everything so far exercises exactly one hardcoded site per
  invocation, no chunking.
- [ ] Phase 4 — `03_classify.py` + `04_classify_blocks.py`
- [ ] Phase 5 — `05_analyze.py` + `label_ui/`
