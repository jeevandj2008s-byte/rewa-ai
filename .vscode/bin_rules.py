"""bin_rules.py - Single source of truth for ReWA.AI bin rules and candidate labels.

Unifies item-to-bin mappings across recycling_logic.py, digital_twin.py, and app1.py.
Loads regional rules (e.g., Bengaluru household waste segregation system).
"""

import json
from pathlib import Path

# 1. Canonical Candidate Labels (used by CLIP zero-shot classification)
CANDIDATE_LABELS = [
    "cardboard box",
    "paper sheet",
    "newspaper",
    "plastic bottle",
    "plastic bag",
    "plastic wrapper",
    "metal can",
    "glass bottle",
    "glass jar",
    "food waste",
    "fruit peel",
    # --- hazardous / medical ---
    "battery",
    "light bulb",
    "medicine",
    "syringe or needle",
    "medical waste",
    "biohazard waste",
    "chemical container",
    "aerosol can",
    "paint can",
    "pesticide or poison",
    # --- catch-all ---
    "other waste",
]

# 2. Canonical Digital Twin Bins (TitleCase)
CANONICAL_BINS = [
    "Plastic",
    "Paper",
    "Metal",
    "Glass",
    "Hazardous",
    "Biodegradable",
]

BIN_IDS = {
    "Plastic": "PLASTIC_01",
    "Paper": "PAPER_01",
    "Metal": "METAL_01",
    "Glass": "GLASS_01",
    "Hazardous": "HAZARDOUS_01",
    "Biodegradable": "BIO_01",
}

# 3. Canonical Item-to-Bin Mapping (Single Source of Truth)
# Every candidate label and common synonyms map to one of the 6 canonical bins
# (or None / "Other" for unclassified / catch-all items).
ITEM_TO_BIN = {
    # Paper
    "cardboard box": "Paper",
    "cardboard": "Paper",
    "paper sheet": "Paper",
    "paper": "Paper",
    "newspaper": "Paper",
    # Plastic
    "plastic bottle": "Plastic",
    "plastic bag": "Plastic",
    "plastic wrapper": "Plastic",
    "plastic container": "Plastic",
    "plastic": "Plastic",
    # Metal
    "metal can": "Metal",
    "aluminium can": "Metal",
    "aluminum can": "Metal",
    "tin can": "Metal",
    "metal": "Metal",
    # Glass
    "glass bottle": "Glass",
    "glass jar": "Glass",
    "glass": "Glass",
    # Biodegradable
    "food waste": "Biodegradable",
    "fruit peel": "Biodegradable",
    "leaves": "Biodegradable",
    # Hazardous
    "battery": "Hazardous",
    "light bulb": "Hazardous",
    "medicine": "Hazardous",
    "syringe or needle": "Hazardous",
    "medical waste": "Hazardous",
    "biohazard waste": "Hazardous",
    "chemical container": "Hazardous",
    "aerosol can": "Hazardous",
    "paint can": "Hazardous",
    "pesticide or poison": "Hazardous",
    # Catch-all
    "other waste": None,
}

# Thresholds
FULL_AT = 90
LOW_SCORE = 0.55  # similarity score threshold below which manual verification is needed


def _load_bengaluru_data():
    """Load Bengaluru municipal rules from JSON."""
    candidates = [
        Path(__file__).resolve().parent / "data" / "bengaluru_rules.json",
        Path(__file__).resolve().parent.parent / "data" / "bengaluru_rules.json",
        Path.cwd() / "data" / "bengaluru_rules.json",
        Path.cwd() / ".vscode" / "data" / "bengaluru_rules.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {"_meta": {}, "rules": {}}


_BENGALURU_DATA = _load_bengaluru_data()


def get_bengaluru_rule(item_label):
    """Return the Bengaluru rule dict for a given label, or a default note."""
    norm = str(item_label).strip().lower()
    rules = _BENGALURU_DATA.get("rules", {})
    if norm in rules:
        return rules[norm]
    # Check simple substring matches
    for key, val in rules.items():
        if key in norm or norm in key:
            return val
    return {
        "bin": ITEM_TO_BIN.get(norm, "Other"),
        "bengaluru_category": "Needs Verification",
        "guidance": "Follow local ward segregation rules.",
        "edge_cases": "Segregation rules may differ across wards. Verify with local municipal guidelines.",
        "source": "NEEDS VERIFICATION",
    }


def get_bengaluru_meta():
    """Return metadata about Bengaluru guidelines."""
    return _BENGALURU_DATA.get("_meta", {
        "city": "Bengaluru",
        "framework": "BBMP 2-Bin 1-Bag Segregation System",
        "ward_variation_note": "Segregation rules vary across wards and apartments. Confirm with your local RWA or ward DWCC.",
    })


def resolve_bin(item_label):
    """Map any item label or description to a canonical bin name."""
    if not item_label:
        return None
    name = str(item_label).strip().lower()
    if name in ITEM_TO_BIN:
        return ITEM_TO_BIN[name]

    # Keyword fallbacks (whole word or substring)
    if any(w in name for w in ["battery", "chemical", "paint", "bulb", "medicine",
                              "syringe", "needle", "sharps", "medical", "biohazard",
                              "aerosol", "pesticide", "poison", "hazardous"]):
        return "Hazardous"
    if any(w in name for w in ["food", "peel", "leaf", "leaves", "vegetable", "fruit"]):
        return "Biodegradable"
    if "glass" in name or "jar" in name:
        return "Glass"
    if any(w in name for w in ["metal", "aluminium", "aluminum", "tin", "steel", "can"]):
        return "Metal"
    if any(w in name for w in ["paper", "cardboard", "newspaper", "carton"]):
        return "Paper"
    if any(w in name for w in ["plastic", "polythene", "bottle", "bag", "wrapper"]):
        return "Plastic"

    return None


def decide_bin(waste_type, score, bins):
    """Standardized bin decision function with similarity score checking."""
    label = str(waste_type).strip().lower()
    bin_name = resolve_bin(label)

    if not bin_name or bin_name not in CANONICAL_BINS:
        return {
            "bin": None,
            "fill_percent": None,
            "is_full": False,
            "needs_verification": True,
            "note": "Unknown waste category. Please verify the item manually.",
        }

    # Support case-insensitive lookup in bins dict
    bins_lookup = {k.capitalize(): v for k, v in (bins or {}).items()}
    bin_info = bins_lookup.get(bin_name, {})
    fill = bin_info.get("fill_percent") or bin_info.get("capacity_percent")

    b_rule = get_bengaluru_rule(label)
    note = b_rule.get("guidance", "")

    return {
        "bin": bin_name,
        "fill_percent": fill,
        "is_full": fill is not None and fill >= FULL_AT,
        "needs_verification": score < LOW_SCORE,
        "note": note,
    }

