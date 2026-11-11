"""Documented valence lexicon for the Kaggle sentiment labels.

The dataset ships 191 distinct emotion labels (after whitespace normalisation),
not the three sentiment classes suggested by its name.  Mapping those labels to
a binary opinion is therefore a modelling choice, and we make it explicit
rather than implicit.

Three sets are defined:

``POSITIVE``
    Labels whose dominant valence is clearly positive.
``NEGATIVE``
    Labels whose dominant valence is clearly negative.
``AMBIGUOUS``
    Labels that are mixed, context dependent or valence-neutral (e.g.
    ``Bittersweet``, ``Nostalgia``, ``Neutral``).  They are *not* mapped to an
    opinion; users whose only posts fall in this set are reported as
    unmapped and excluded from data-initialised runs.

The classification follows the valence dimension of standard emotion models
(positive/negative affect) and was fixed before looking at any simulation
output.  ``docs/DATA.md`` reports the class counts and a sensitivity analysis
against the paper's original rule (``Sentiment == "Positive" -> 1``).
"""

from __future__ import annotations

import re
from typing import Optional

POSITIVE = frozenset(
    {
        "positive", "joy", "excitement", "contentment", "gratitude", "serenity",
        "happy", "awe", "hopeful", "acceptance", "pride", "elation", "euphoria",
        "enthusiasm", "determination", "playful", "inspiration", "happiness",
        "hope", "empowerment", "inspired", "admiration", "calmness", "compassion",
        "tenderness", "fulfillment", "reverence", "proud", "grateful",
        "compassionate", "thrill", "enchantment", "love", "amusement",
        "anticipation", "kind", "empathetic", "confident", "satisfaction",
        "accomplishment", "harmony", "creativity", "wonder", "adventure",
        "enjoyment", "affection", "adoration", "zest", "whimsy", "radiance",
        "rejuvenation", "coziness", "resilience", "exploration", "captivation",
        "tranquility", "overjoyed", "motivation", "joyfulreunion", "blessed",
        "appreciation", "confidence", "wonderment", "optimism", "playfuljoy",
        "mindfulness", "dreamchaser", "elegance", "melodic", "festivejoy",
        "innerjourney", "freedom", "dazzle", "adrenaline", "artisticburst",
        "culinaryodyssey", "spark", "marvel", "positivity", "kindness",
        "friendship", "success", "amazement", "romance", "grandeur", "energy",
        "celebration", "charm", "ecstasy", "colorful", "connection", "iconic",
        "engagement", "touched", "triumph", "heartwarming", "sympathy",
        "renewed effort", "breakthrough", "joy in baking", "envisioning history",
        "imagination", "vibrancy", "mesmerizing", "culinary adventure",
        "winter magic", "thrilling journey", "nature's beauty",
        "celestial wonder", "creative inspiration", "runway creativity",
        "ocean's freedom", "relief", "solace", "free-spirited",
    }
)

NEGATIVE = frozenset(
    {
        "despair", "grief", "loneliness", "sad", "embarrassed", "frustration",
        "regret", "hate", "bad", "disgust", "bitterness", "frustrated",
        "betrayal", "negative", "boredom", "desolation", "bitter", "shame",
        "jealousy", "resentment", "fearful", "jealous", "devastated", "envious",
        "dismissive", "heartbreak", "anger", "fear", "sadness", "disappointed",
        "anxiety", "intimidation", "helplessness", "envy", "yearning",
        "apprehensive", "isolation", "disappointment", "sorrow", "loss",
        "suffering", "emotionalstorm", "lostlove", "exhaustion", "darkness",
        "desperation", "ruins", "heartache", "suspense", "obstacle",
        "pressure", "miscalculation", "challenge", "melancholy",
    }
)

AMBIGUOUS = frozenset(
    {
        "neutral", "curiosity", "confusion", "surprise", "nostalgia",
        "indifference", "numbness", "ambivalence", "arousal", "overwhelmed",
        "reflection", "contemplation", "emotion", "mischievous", "bittersweet",
        "intrigue", "pensive", "immersion", "solitude", "hypnotic", "journey",
        "whispers of the past",
    }
)

_ALL = POSITIVE | NEGATIVE | AMBIGUOUS
_OVERLAP = (POSITIVE & NEGATIVE) | (POSITIVE & AMBIGUOUS) | (NEGATIVE & AMBIGUOUS)
if _OVERLAP:  # pragma: no cover - import-time contract
    raise RuntimeError(f"lexicon sets overlap: {sorted(_OVERLAP)}")


def normalize_label(label: object) -> str:
    """Whitespace-normalised, case-folded label used as lexicon key."""
    return re.sub(r"\s+", " ", str(label)).strip().casefold()


def classify(label: object) -> Optional[int]:
    """Return 1 (positive), 0 (negative) or ``None`` (ambiguous/unknown)."""
    key = normalize_label(label)
    if key in POSITIVE:
        return 1
    if key in NEGATIVE:
        return 0
    return None
