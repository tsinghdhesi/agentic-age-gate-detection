"""The model prompts, in ONE place.

DETECTION_PROMPT is a FROZEN measurement instrument. It was signed off by the
researcher and must not be edited inline anywhere else. Its sha256 is recorded
per run (runs.detection_prompt_hash) and per detection row, so any change to the
wording is detectable downstream. If the instrument must ever change, change it
HERE, deliberately, and treat results under the new hash as a separate measure.

The navigation prompt is NOT a frozen instrument — it is operational plumbing
for reaching checkpoints and may be iterated.
"""

from __future__ import annotations

import hashlib

# --------------------------------------------------------------------------- #
# FROZEN — age-gate detection instrument. Do not edit without intent.
# --------------------------------------------------------------------------- #
DETECTION_PROMPT = (
    "You are looking at a single screenshot of a webpage. Your only task is to "
    "decide whether an age-verification gate is present.\n"
    "\n"
    "An age-verification gate is any overlay, modal, interstitial, or page whose "
    "purpose is to confirm the visitor's age before granting access. Structurally "
    "it usually appears as a layer that covers or blocks the underlying page "
    "content and prevents the visitor from accessing or interacting with the page "
    "until they perform an action — confirming they are old enough, choosing "
    "Yes/No, or entering a date of birth. Recognize it by this purpose and "
    "structure REGARDLESS OF THE LANGUAGE IT IS WRITTEN IN. MAKE SURE YOUR OUTPUT"
    "IS LANGUAGE INVARIANT — for example:"
    "\"Are you 21 or older?\", \"You must be 18+ to enter\", \"Please verify your "
    "age\", a Yes/No age confirmation, or a date-of-birth entry required to "
    "proceed.\n"
    "\n"
    "Do not count any of the following as an age gate: cookie/consent banners, "
    "newsletter or email-signup popups, country/region/language selectors that do "
    "not ask about age, login or account forms, generic promotional or discount "
    "modals, shipping notices, sensitive content warnings, or bot-detection / security block pages.\n"
    "\n"
    "Answer with exactly one character and nothing else:\n"
    "1 if an age-verification gate is present, or 0 if it is not."
)

DETECTION_PROMPT_HASH = hashlib.sha256(DETECTION_PROMPT.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Navigation (operational, iterable). Model sees the VIEWPORT screenshot and the
# checkpoint goal; coordinates are on the 1000x1000 canvas (see common.coords).
# --------------------------------------------------------------------------- #
def navigation_prompt(goal: str) -> str:
    return (
        "You control a web browser to accomplish one goal. You are shown a "
        "screenshot of the current browser VIEWPORT. A coordinate grid of "
        "1000x1000 is overlaid on the image: x runs 0 (left) to 1000 (right), "
        "y runs 0 (top) to 1000 (bottom).\n"
        "\n"
        f"GOAL: {goal}\n"
        "\n"
        "Decide the SINGLE best next action toward the goal and reply with JSON "
        "only, no prose:\n"
        '{"action": "click|type|scroll|done|stuck", "box": [x1,y1,x2,y2], "text": ""}\n'
        "\n"
        "Rules:\n"
        '- "click": box is the element to click (1000x1000 coords).\n'
        '- "type": box is the input field; text is what to type.\n'
        '- "scroll": text is "down" or "up"; box is ignored.\n'
        '- "done": the goal is already satisfied / visible on screen.\n'
        '- "stuck": nothing visible advances the goal.\n'
        "Only act on what is currently visible in the viewport."
    )


# JSON schema handed to Ollama `format=` so navigation output is always valid
# structured JSON. Only `action` is required; box/text are validated in code.
NAV_FORMAT = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["click", "type", "scroll", "done", "stuck"],
        },
        "box": {"type": "array", "items": {"type": "number"}},
        "text": {"type": "string"},
    },
    "required": ["action"],
}
