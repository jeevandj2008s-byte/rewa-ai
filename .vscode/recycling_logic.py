"""recycling_logic.py - Bin decision logic for ReWA.AI.

Imports unified rules from bin_rules.py (Single Source of Truth).
"""

from bin_rules import (
    CANDIDATE_LABELS,
    CANONICAL_BINS,
    ITEM_TO_BIN,
    FULL_AT,
    LOW_SCORE,
    decide_bin,
    get_bengaluru_rule,
    resolve_bin,
)

# Backward-compatibility aliases
CATEGORY_TO_BIN = {k: (v.lower() if v else None) for k, v in ITEM_TO_BIN.items()}

SOFT = "Soft plastic is often NOT accepted in regular plastic bins. Check local rules."
NOTES = {
    "plastic bag": SOFT,
    "plastic wrapper": SOFT,
    "food waste": "Organic waste: compost or green bin per Bengaluru BBMP rules.",
    "other waste": "No matching bin in this system. Check the item's label and local rules.",
}


def twin_to_shared(twin_bins):
    return {
        name.lower(): {
            "fill_percent": data["capacity_percent"],
            "item_count": data["item_count"],
            "source": "simulated",
        }
        for name, data in twin_bins.items()
    }
