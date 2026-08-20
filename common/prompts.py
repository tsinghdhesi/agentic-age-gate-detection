"""The model prompts, in ONE place.

DETECTION_PROMPT and RELEVANCE_PROMPT are FROZEN measurement instruments.
They were signed off by the researcher and must not be edited inline
anywhere else. Their sha256 hashes are recorded per row downstream (per
detection row for DETECTION_PROMPT_HASH; per sites row for
RELEVANCE_PROMPT_HASH), so any change to the wording is detectable. If an
instrument must ever change, change it HERE, deliberately, and treat results
under the new hash as a separate measure. RELEVANCE_PROMPT determines the
sampling frame itself — at least as consequential for external validity as
the detection prompt — and gets the same treatment, not a lighter one.

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
# FROZEN — frame-relevance instrument. Do not edit without intent. Determines
# the sampling frame for 01b_screen_relevance.py / common.vlm.classify_relevance().
# --------------------------------------------------------------------------- #
RELEVANCE_PROMPT = (
    "You are looking at extracted text from a website's home page: its title, "
    "meta description, main heading, and a portion of the visible body text. "
    "Decide what kind of site this is, specifically with respect to nicotine "
    "and e-cigarette products — vapes, e-liquids, disposables, pods, "
    "cigarettes, cigars, pipe tobacco, snus, nicotine pouches, hookah/shisha "
    "tobacco.\n"
    "\n"
    "Judge the SITE'S PURPOSE from this text REGARDLESS OF THE LANGUAGE IT IS "
    "WRITTEN IN — a German site about \"E-Zigaretten\" or a Bulgarian site "
    "about \"никотин\" must be judged by the same standard as an English one. "
    "MAKE SURE YOUR OUTPUT IS LANGUAGE INVARIANT.\n"
    "\n"
    "Choose exactly one category:\n"
    "- \"retailer\": an online shop that sells nicotine or e-cigarette "
    "products directly to consumers\n"
    "- \"brand_manufacturer\": the official site of a nicotine or "
    "e-cigarette brand or manufacturer, whether or not it sells directly\n"
    "- \"marketplace\": a general marketplace or classifieds site that "
    "includes nicotine or e-cigarette listings among other unrelated "
    "products\n"
    "- \"informational\": news, blog, forum, review, research, or advocacy "
    "content about nicotine or vaping that is not itself selling anything\n"
    "- \"unrelated\": the page has nothing to do with nicotine or "
    "e-cigarette products\n"
    "\n"
    "Be careful: a product or brand NAME that merely contains a "
    "nicotine-adjacent WORD is not evidence the site sells nicotine products "
    "— for example a perfume called \"Tobacco Vanille\" (tobacco is a common "
    "fragrance note), a restaurant named \"Smoke & Grill\", or a BBQ shop "
    "selling \"smoked\" meats. Judge the site's actual business and content, "
    "never an incidental word match.\n"
    "\n"
    "If the page shows only a placeholder — \"under construction\", \"coming "
    "soon\", a maintenance notice, a parked-domain page, or is otherwise "
    "empty of real content — classify it as \"unrelated\", even if the "
    "domain name or title suggests a nicotine-related business. Judge only "
    "what the page actually shows right now, never what the domain name "
    "implies about future or intended content.\n"
    "\n"
    "Reply with JSON only, no prose:\n"
    '{"category": "retailer|brand_manufacturer|marketplace|informational|unrelated"}'
)

RELEVANCE_PROMPT_HASH = hashlib.sha256(RELEVANCE_PROMPT.encode("utf-8")).hexdigest()

# JSON schema handed to Ollama `format=` for classify_relevance() — mirrors
# NAV_FORMAT's constrained-generation pattern below.
RELEVANCE_FORMAT = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["retailer", "brand_manufacturer", "marketplace", "informational", "unrelated"],
        },
    },
    "required": ["category"],
}


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
