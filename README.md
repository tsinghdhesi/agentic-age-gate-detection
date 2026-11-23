# Agentic Age-Gate Detection

Local, serial research pipeline that crawls regulated-product sites (nicotine /
alcohol / gambling) through the purchase funnel and detects, per checkpoint,
whether an **age gate** is present. Vision-language inference runs **fully
locally** via Qwen3-VL on Ollama — **no cloud calls, ever**.

> Safety by construction: never completes a purchase, never solves CAPTCHAs,
> serial execution only. See [CLAUDE.md](CLAUDE.md) for the full invariant list
> and design decisions.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
# Ollama must be running with the Qwen3-VL models pulled:
#   ollama pull qwen3-vl:8b qwen3-vl:30b qwen3-vl:32b
```

## Layout
- `common/` — shared library (schema, config, coords, VLM client, …). Imported, never run.
- `config/` — `global.yaml` + per-industry checkpoint configs.
- `scripts/` — numbered pipeline stages: `01_build_queue` → `01b_screen_relevance` → `01c_draw_sample` → `02_collect` → `03_classify` → `04_classify_blocks` → `05_analyze`.
  Plus two unnumbered scripts: `chunk_work.py` (resumable work chunks, used by several stages) and `02_collect_browseruse.py` (a second navigation engine, parallel to `02_collect`).
- `spike/` — Day-1 grounding / detection validation.
- `label_ui/` — minimal local manual-review page.
- `data/` — seed inputs and the hand-labeled relevance gold set (tracked). Run outputs (screenshots / DOM / HTML / logs / SQLite) and raw gold-set fetches are generated and gitignored.
- `tests/` — pytest suite.

## Verify the build
```bash
python -m pytest -q            # full unit test suite
python -c "from common import db; c=db.connect(); db.init_db(c); print('schema ok')"
```

Status: see the build checklist at the bottom of [CLAUDE.md](CLAUDE.md).
