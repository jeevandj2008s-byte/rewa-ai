CATEGORY_TO_BIN = {
    "cardboard box": "paper",
    "paper sheet": "paper",
    "newspaper": "paper",
    "plastic bottle": "plastic",
    "plastic bag": "plastic",
    "plastic wrapper": "plastic",
    "metal can": "metal",
    "glass bottle": "glass",
    "glass jar": "glass",
    "food waste": None,    # no bin in the Digital Twin
    "other waste": None,
}

SOFT = "Soft plastic is often NOT accepted in regular plastic bins. Check local rules."
NOTES = {
    "plastic bag": SOFT,
    "plastic wrapper": SOFT,
    "food waste": "No recycling bin in this system. Compost or general waste, per local rules.",
    "other waste": "No matching bin in this system. Check the item's label and local rules.",
}

FULL_AT = 90
LOW_SCORE = 0.55   # same threshold your app uses

def decide_bin(waste_type, score, bins):
    label = waste_type.lower()
    if label not in CATEGORY_TO_BIN:
        return {"bin": None, "fill_percent": None, "is_full": False,
                "needs_verification": True,
                "note": "Unknown category. Verify the item manually."}
    bin_name = CATEGORY_TO_BIN[label]
    fill = bins.get(bin_name, {}).get("fill_percent") if bin_name else None
    return {
        "bin": bin_name,
        "fill_percent": fill,
        "is_full": fill is not None and fill >= FULL_AT,
        "needs_verification": score < LOW_SCORE,
        "note": NOTES.get(label, ""),
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