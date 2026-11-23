"""Canonical + debug artifact paths for the funnel crawl (02_collect*.py) —
the ONE place that builds these, so a collection script and a future
classification pass can never disagree about where an image lives.

Canonical screenshots nest under run_id:
    data/screenshots/<run_id>/<site_id>/<checkpoint>_{fullpage,viewport}.png

This is a deliberate AMENDMENT to CLAUDE.md's original locked decision
("site_id alone is the screenshot folder name") — added when
02_collect_browseruse.py was built. A `--retry-blocked` re-attempt under a
NEW run_id must never silently overwrite the first attempt's images on disk
while that attempt's `observations` row still points at the (now wrong)
path. `data/html/<site_id>.html.gz` (01b's fetch store) is unaffected —
that is inherently a one-per-site, no-run-concept artifact.

Debug artifacts (per-agent-step screenshots + action trail JSON, NEVER read
by the classification pass) live one level deeper:
    data/screenshots/<run_id>/<site_id>/debug/<checkpoint>/step_<NN>.png
    data/screenshots/<run_id>/<site_id>/debug/<checkpoint>/trail.json
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = REPO_ROOT / "data" / "screenshots"


def site_dir(run_id: str, site_id: str) -> Path:
    return SCREENSHOTS_DIR / run_id / site_id


def fullpage_path(run_id: str, site_id: str, checkpoint: str) -> Path:
    return site_dir(run_id, site_id) / f"{checkpoint}_fullpage.png"


def viewport_path(run_id: str, site_id: str, checkpoint: str) -> Path:
    return site_dir(run_id, site_id) / f"{checkpoint}_viewport.png"


def debug_dir(run_id: str, site_id: str, checkpoint: str) -> Path:
    return site_dir(run_id, site_id) / "debug" / checkpoint


def debug_step_screenshot_path(run_id: str, site_id: str, checkpoint: str, step: int) -> Path:
    return debug_dir(run_id, site_id, checkpoint) / f"step_{step:02d}.png"


def debug_trail_json_path(run_id: str, site_id: str, checkpoint: str) -> Path:
    return debug_dir(run_id, site_id, checkpoint) / "trail.json"
