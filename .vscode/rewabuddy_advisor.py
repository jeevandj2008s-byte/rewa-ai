import os
import time
import requests

from google import genai
from recycling_logic import decide_bin


# ============================================================
# SMART RECYCLING ASSISTANT — CONFIGURATION
# ============================================================

# NOTE: Check the exact model name in Google AI Studio.
# A wrong name makes Gemini always fail and fall back to Groq.
GEMINI_MODELS = [
    "gemini-3.8-flash",
]

GROQ_MODEL = "openai/gpt-oss-120b"

OPENROUTER_MODEL = "openrouter/free"

REQUEST_TIMEOUT = 30

MIN_CONFIDENCE_FOR_AUTOMATIC_SORTING = 0.55

# A bin at or above this simulated fill level counts as "nearly full"
# (same threshold the dashboard uses for its red warning).
BIN_FULL_THRESHOLD = 90


# ============================================================
# MATERIAL CLASSIFICATION (decided by Python, not by the AI)
# Matches the problem statement: recyclable, reusable,
# biodegradable, hazardous
# ============================================================

HAZARDOUS_KEYWORDS = [
    "battery", "e-waste", "electronic", "chemical", "bulb",
    "medicine", "paint", "aerosol", "syringe", "tube light",
    "cfl", "pesticide", "thermometer", "toner",
]

BIODEGRADABLE_KEYWORDS = [
    "food", "leaf", "leaves", "fruit", "vegetable", "organic",
    "peel", "garden", "eggshell", "tea leaves", "coffee grounds",
    "flower", "grass",
]

# If the label mentions packaging, it is NOT organic waste
# (e.g. "food wrapper" is packaging, not food).
PACKAGING_WORDS = ["plastic", "wrapper", "packet", "packaging", "bottle", "can"]


def classify_material(waste_type):
    """Return 'hazardous', 'biodegradable', or 'recyclable and reusable'."""
    label = str(waste_type).lower()

    if any(word in label for word in HAZARDOUS_KEYWORDS):
        return "hazardous"

    if any(word in label for word in BIODEGRADABLE_KEYWORDS):
        if not any(word in label for word in PACKAGING_WORDS):
            return "biodegradable"

    return "recyclable and reusable"


# ============================================================
# BIN DECISION FOR HAZARDOUS / BIODEGRADABLE ITEMS
# (decide_bin from recycling_logic.py still handles the
#  plastic / paper / metal / glass items)
# ============================================================

SPECIAL_BINS = {
    "hazardous": "Hazardous",
    "biodegradable": "Biodegradable",
}


def decide_special_bin(material_class, confidence, bin_status):
    """Python-only bin decision for hazardous and biodegradable items."""
    bin_name = SPECIAL_BINS[material_class]

    # bin_status keys are lowercase ("hazardous"), but be tolerant
    status_lookup = {
        str(key).lower(): value
        for key, value in (bin_status or {}).items()
    }
    info = status_lookup.get(material_class)

    fill = None
    if isinstance(info, dict):
        fill = info.get("fill_percent")

    is_full = fill is not None and fill >= BIN_FULL_THRESHOLD

    if material_class == "hazardous":
        note = (
            "Hazardous items need special handling. Keep them "
            "separate from ordinary waste and recycling."
        )
    else:
        note = (
            "Keep organic waste separate from dry recyclables "
            "so it can be composted."
        )

    return {
        "bin": bin_name,
        "fill_percent": fill,
        "is_full": is_full,
        "needs_verification": confidence < MIN_CONFIDENCE_FOR_AUTOMATIC_SORTING,
        "note": note,
    }


# ============================================================
# REUSE KNOWLEDGE BASE (written by us, so the AI never has to
# invent reuse or safety advice on its own)
# ============================================================

# --- Recyclable materials: 3 safe, cheap reuse ideas per item ---
ITEM_REUSE = {
    "plastic bottle": [
        "Cut the bottle and use the bottom half as a pen stand or a small planter.",
        "Poke holes in the cap and use it as a watering can for plants.",
        "Use a clean bottle to store dry items such as rice, pulses or loose stationery.",
    ],
    "plastic bag": [
        "Reuse it as a bin liner for small dustbins.",
        "Carry it again for groceries or keep it in your bag for wet clothes.",
        "Use it to protect books or shoes inside your backpack on rainy days.",
    ],
    "plastic wrapper": [
        "Cut clean wrappers into strips for craft projects or collages.",
        "Use a clean wrapper to wrap small items when packing a parcel.",
        "Collect clean wrappers for school or college art projects.",
    ],
    "plastic container": [
        "Wash it well and use it to store snacks, screws, buttons or stationery.",
        "Use it as a tray under a plant pot.",
        "Use it as a drawer organiser for small items.",
    ],
    "cardboard box": [
        "Flatten it and reuse it for packing, or keep it for moving house.",
        "Cut it into drawer dividers or a desk organiser.",
        "Use it as a base for crafts, models or a college project.",
    ],
    "paper sheet": [
        "Use the blank side for rough work or notes.",
        "Cut clean sheets into small notepads or sticky-note style pads.",
        "Use it for origami, paper crafts or gift wrapping.",
    ],
    "newspaper": [
        "Use it to wrap fragile items or gifts.",
        "Use it to line drawers or shelves.",
        "Crumple it to clean glass and mirrors, or use it to cover surfaces while painting.",
    ],
    "metal can": [
        "Wash it, cover any sharp edge with tape, and use it as a pen holder.",
        "Use it as a small planter after making drainage holes carefully.",
        "Decorate it and use it as a desk organiser for small tools.",
    ],
    "glass bottle": [
        "Wash it and use it as a water bottle for the fridge or a flower vase.",
        "Use it to store homemade items like pickles or juice, only if it is properly cleaned.",
        "Decorate it as a lamp base or a desk decoration.",
    ],
    "glass jar": [
        "Wash it and use it to store spices, pulses or snacks.",
        "Use it as a pen holder or a small planter for succulents.",
        "Use it as a candle holder or decoration.",
    ],
}

# --- Fallback by material keyword when the exact label is unknown ---
CATEGORY_REUSE = [
    (["plastic"], ITEM_REUSE["plastic container"]),
    (["cardboard", "carton", "box"], ITEM_REUSE["cardboard box"]),
    (["paper"], ITEM_REUSE["paper sheet"]),
    (["metal", "tin", "can", "aluminium", "aluminum"], ITEM_REUSE["metal can"]),
    (["glass", "jar"], ITEM_REUSE["glass jar"]),
]

GENERIC_REUSE = [
    "Check whether the item can be reused for storage or as a container.",
    "Donate it if it is still in good condition.",
    "Repurpose it for a craft or college project before discarding.",
]

# --- Biodegradable items: composting and soil-friendly ideas ---
BIODEGRADABLE_REUSE = {
    "food waste": [
        "Add vegetable and fruit scraps to a home compost pit or a compost bin.",
        "Try a small kitchen compost bucket if you live in a hostel or flat.",
        "Take it to a community compost point if your area has one. Do not mix it with plastic.",
    ],
    "fruit peel": [
        "Chop banana or fruit peels and bury them in plant soil to enrich it.",
        "Make an eco-enzyme cleaner by fermenting fruit peels with jaggery and water.",
        "Add the peels to a compost bin or bucket.",
    ],
    "leaves": [
        "Use dry leaves as mulch around plants to keep the soil moist.",
        "Add dry leaves to a compost pile to balance wet kitchen waste.",
        "Use clean dry leaves for pressed-leaf art or crafts.",
    ],
}

BIODEGRADABLE_DEFAULT = [
    "Compost it at home or in a community compost bin.",
    "Bury small amounts in plant soil so they break down naturally.",
    "Keep it separate from plastic, metal and other dry waste.",
]

# --- Hazardous items: NO DIY reuse, only safe handling ---
HAZARDOUS_GUIDE = {
    "battery": {
        "reuse_line": (
            "Do not reuse or open used batteries. Switching to rechargeable "
            "batteries in future reduces battery waste."
        ),
        "handling": [
            "Never throw batteries in the normal dustbin or in a fire.",
            "Tape the ends of button or lithium cells and keep them dry.",
            "If a battery is swollen, leaking or hot, do not charge or puncture it. "
            "Place it in a non-metal container and take it to a collection point.",
        ],
    },
    "light bulb": {
        "reuse_line": (
            "Do not reuse a damaged or used bulb. A working bulb can be used "
            "in another fitting."
        ),
        "handling": [
            "CFL and tube lights contain mercury, so do not put them in normal waste.",
            "Wrap an intact bulb in paper or its box before taking it to a collection point.",
            "If one breaks, ventilate the room and pick up pieces with cardboard "
            "and gloves. Do not use a vacuum cleaner.",
        ],
    },
    "medicine": {
        "reuse_line": (
            "Do not reuse or share leftover or expired medicines. Return them "
            "through a take-back option."
        ),
        "handling": [
            "Return unused or expired medicines to a pharmacy or take-back point.",
            "Do not flush medicines unless the label specifically says so.",
            "Keep them in the original packaging, away from children and pets, "
            "until you dispose of them.",
        ],
    },
    "default": {
        "reuse_line": (
            "This item is hazardous, so do not try to reuse it yourself. "
            "Take it to a special collection point."
        ),
        "handling": [
            "Keep it separate from ordinary waste and recycling.",
            "Use an authorised hazardous-waste or e-waste collection point.",
            "Keep it in its original container and away from heat, children and pets.",
        ],
    },
}


def get_reuse_info(waste_type, material_class):
    """
    Return reuse ideas and safety tips picked by Python.

    {
        "ideas":    [...],  # reuse / composting ideas (empty for hazardous)
        "reuse_line": str or None,  # short line for hazardous items
        "handling": [...],  # safe-handling tips (hazardous only)
    }
    """
    label = str(waste_type).strip().lower()

    if material_class == "hazardous":
        guide = HAZARDOUS_GUIDE["default"]
        for key in ("battery", "light bulb", "medicine"):
            if key.split()[-1] in label:
                guide = HAZARDOUS_GUIDE[key]
                break
        if "tube" in label or "cfl" in label:
            guide = HAZARDOUS_GUIDE["light bulb"]
        return {
            "ideas": [],
            "reuse_line": guide["reuse_line"],
            "handling": guide["handling"],
        }

    if material_class == "biodegradable":
        ideas = BIODEGRADABLE_REUSE.get(label)
        if ideas is None:
            if "leaf" in label or "leaves" in label:
                ideas = BIODEGRADABLE_REUSE["leaves"]
            elif "peel" in label or "fruit" in label:
                ideas = BIODEGRADABLE_REUSE["fruit peel"]
            elif "food" in label or "vegetable" in label:
                ideas = BIODEGRADABLE_REUSE["food waste"]
            else:
                ideas = BIODEGRADABLE_DEFAULT
        return {"ideas": ideas, "reuse_line": None, "handling": []}

    # Recyclable and reusable
    if label in ITEM_REUSE:
        ideas = ITEM_REUSE[label]
    else:
        ideas = GENERIC_REUSE
        for words, candidate in CATEGORY_REUSE:
            if any(word in label for word in words):
                ideas = candidate
                break

    return {"ideas": ideas, "reuse_line": None, "handling": []}


# ============================================================
# PROVIDER 1 — GEMINI
# ============================================================

def try_gemini(prompt):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    last_error = None

    for model in GEMINI_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text, f"Gemini ({model})"

            except Exception as error:
                last_error = error

                # Retry temporary server errors once.
                error_text = str(error)

                temporary_error = any(
                    code in error_text
                    for code in ["429", "500", "502", "503", "504"]
                )

                if temporary_error and attempt == 0:
                    time.sleep(2)
                    continue

                break

    raise RuntimeError(f"Gemini failed: {last_error}")


# ============================================================
# SHARED OPENAI-COMPATIBLE API CALL
# Used by Groq and OpenRouter
# ============================================================

def call_openai_style(url, key_env, model, prompt, label):
    api_key = os.getenv(key_env)

    if not api_key:
        raise RuntimeError(f"{key_env} is not set.")

    if not model:
        raise RuntimeError(f"{label} model is not configured.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # OpenRouter supports these optional headers.
    if "openrouter.ai" in url:
        headers["HTTP-Referer"] = "http://localhost:8051"
        headers["X-Title"] = "EcoSort AI"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are EcoSort AI, a careful recycling and "
                    "reuse assistant. "
                    "Use only the facts supplied in the prompt. "
                    "Never invent bin levels, measurements, or facts. "
                    "Explain uncertainty clearly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.3,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"{label} returned an unexpected response."
        ) from error

    if isinstance(answer, list):
        answer = "\n".join(
            part.get("text", "")
            for part in answer
            if isinstance(part, dict)
        )

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"{label} returned an empty response.")

    return answer.strip(), f"{label} ({model})"


# ============================================================
# PROVIDER 2 — GROQ
# ============================================================

def try_groq(prompt):
    return call_openai_style(
        url="https://api.groq.com/openai/v1/chat/completions",
        key_env="GROQ_API_KEY",
        model=GROQ_MODEL,
        prompt=prompt,
        label="Groq",
    )


# ============================================================
# PROVIDER 3 — OPENROUTER
# ============================================================

def try_openrouter(prompt):
    return call_openai_style(
        url="https://openrouter.ai/api/v1/chat/completions",
        key_env="OPENROUTER_API_KEY",
        model=OPENROUTER_MODEL,
        prompt=prompt,
        label="OpenRouter",
    )


# ============================================================
# AI PROVIDER FALLBACK
# Gemini → Groq → OpenRouter
# ============================================================

def ask_ai(prompt):
    errors = []

    providers = [
        try_gemini,
        try_groq,
        try_openrouter,
    ]

    for provider in providers:
        try:
            answer, source = provider(prompt)

            print(f"EcoSort AI response generated by {source}")

            return answer, source

        except Exception as error:
            message = str(error)

            # Avoid printing API keys or request headers.
            print(
                f"{provider.__name__} failed: "
                f"{message[:250]}"
            )

            errors.append(
                f"{provider.__name__}: {message[:250]}"
            )

    raise RuntimeError(
        "All configured AI providers failed. "
        + " | ".join(errors)
    )


# ============================================================
# PYTHON-ONLY FALLBACK
# Used when all AI providers fail
# ============================================================

def offline_advice(waste_type, decision, material_class="recyclable and reusable"):
    reuse_info = get_reuse_info(waste_type, material_class)

    lines = [
        "## ♻️ Recycling Advice",
        f"Detected item: **{waste_type}** (class: {material_class}).",
    ]

    if decision.get("needs_verification", True):
        lines.append(
            "Please verify the material before deciding "
            "which bin to use."
        )

    if decision.get("note"):
        lines.append(decision["note"])

    lines.append("## 🗑️ Recommended Bin")

    bin_name = decision.get("bin")

    if bin_name is None:
        lines.append(
            "No matching bin is available in this system."
        )
    else:
        fill = decision.get("fill_percent")

        if fill is None:
            lines.append(
                f"The suggested bin is **{bin_name}**."
            )
        else:
            lines.append(
                f"The suggested bin is **{bin_name}** "
                f"(simulated fill: {fill}%)."
            )

        if decision.get("is_full"):
            lines.append(
                "This bin is nearly full. Consider another "
                "appropriate collection point or report it."
            )

    # Reuse section depends on the material class.
    lines.append("## 🔁 Reuse Ideas")

    if material_class == "hazardous":
        lines.append(reuse_info["reuse_line"])
    elif material_class == "biodegradable":
        lines.append(
            "This item is biodegradable. Compost it instead of "
            "throwing it away.\n"
            + "\n".join(f"- {idea}" for idea in reuse_info["ideas"])
        )
    else:
        lines.append(
            "\n".join(f"- {idea}" for idea in reuse_info["ideas"])
        )

    lines.append("## 💡 Smart Suggestion")

    if material_class == "hazardous":
        tips = reuse_info["handling"]
    else:
        tips = [
            "Check the item's material marking.",
            "Keep recyclable materials clean when required.",
            "Follow your local recycling rules.",
        ]

    lines.append("\n".join(f"- {tip}" for tip in tips))

    return "\n\n".join(lines)


# ============================================================
# MAIN RECYCLING ADVISOR
# Compatible with the CURRENT app1.py call:
#
# generate_recycling_advice(
#     waste_type=best_label,
#     confidence=best_score,
#     bin_status=bin_status
# )
# ============================================================

def generate_recycling_advice(
    waste_type,
    confidence=None,
    bin_status=None,
    score=None,
    bins=None,
):
    """
    Generate recycling AND reuse advice using AI providers
    with a Python-only fallback.

    Supports both calling styles:

    1. Current app1.py:
       waste_type, confidence, bin_status

    2. Earlier advisor interface:
       waste_type, score, bins
    """

    # Support the older argument names as well.
    if score is not None:
        confidence = score

    if bins is not None:
        bin_status = bins

    if confidence is None:
        confidence = 0.0

    if bin_status is None:
        bin_status = {}

    # The CLIP score is a similarity score, not a
    # calibrated probability of correctness.
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))

    # Python decides the material class; the AI only explains it.
    material_class = classify_material(waste_type)

    # Python selects the bin.
    if material_class in SPECIAL_BINS:
        decision = decide_special_bin(
            material_class,
            confidence,
            bin_status,
        )
    else:
        decision = decide_bin(
            waste_type,
            confidence,
            bin_status,
        )

    # Python also selects the reuse ideas / safety tips.
    reuse_info = get_reuse_info(waste_type, material_class)

    if material_class == "hazardous":
        reuse_block = (
            "- Reuse line (hazardous, no DIY reuse): "
            f"{reuse_info['reuse_line']}\n"
            "- Safe handling tips:\n"
            + "\n".join(f"  * {tip}" for tip in reuse_info["handling"])
        )
    else:
        reuse_block = "\n".join(
            f"- {idea}" for idea in reuse_info["ideas"]
        )

    prompt = f"""
You are EcoSort AI, a friendly recycling and reuse assistant
for first-year college students.

Explain the following checked facts in simple English.

FACTS FROM PYTHON:
- Detected image label: {waste_type}
- Material class: {material_class}
- CLIP similarity score: {confidence:.3f}
- This score is NOT a calibrated probability.
- Suggested bin: {decision.get("bin")}
- Bin fill level: {decision.get("fill_percent")}
- Bin readings are simulated, not measured by real sensors.
- Bin is nearly full: {decision.get("is_full")}
- Material verification needed: {decision.get("needs_verification")}
- Additional note: {decision.get("note")}

REUSE / SAFETY INFORMATION FROM PYTHON:
{reuse_block}

RULES:
1. Do not claim the image classification is certainly correct.
2. If verification is needed, tell the user to check the material.
3. Use only the facts provided above.
4. Never invent a bin level or sensor reading.
5. If a value is unavailable, say that it is unavailable.
6. If the suggested bin is nearly full, recommend reporting it
   or finding another appropriate collection point.
7. Mention that recycling rules vary by location.
8. Give practical advice suitable for a college student.
9. Do not change the bin selected by Python.
10. Under Reuse Ideas, use the ideas supplied by Python above.
    You may reword them in simple English, but do not add
    unsafe, expensive or tool-heavy ideas.
11. Reuse ideas must be safe, cheap, and doable by a college
    student with no special tools.
12. If the material class is hazardous (battery, e-waste, bulbs,
    medicine, chemicals), do NOT suggest DIY reuse. Under Reuse
    Ideas write only the one short reuse line supplied by Python,
    and use the safe handling tips for Smart Suggestion.
13. If the material class is biodegradable (food, leaves, peels),
    present the composting ideas supplied by Python as the reuse
    ideas.

Use exactly these four headings:

## ♻️ Recycling Advice

## 🗑️ Recommended Bin

## 🔁 Reuse Ideas

## 💡 Smart Suggestion

Under Reuse Ideas, give exactly three short ideas
(except for hazardous items, see rule 12).
Under Smart Suggestion, provide exactly three short tips.
"""

    try:
        answer, source = ask_ai(prompt)

        return (
            f"{answer}\n\n"
            f"*Answered by: {source}*"
        )

    except Exception as error:
        print(
            "All AI providers failed; using Python fallback:",
            str(error)[:300],
        )

        return (
            offline_advice(waste_type, decision, material_class)
            + "\n\n"
            "*AI providers are unavailable. "
            "This answer was generated using Python rules.*"
        )


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    test_bins = {
        "plastic": {"fill_percent": 45, "source": "simulated"},
        "paper": {"fill_percent": 70, "source": "simulated"},
        "metal": {"fill_percent": 25, "source": "simulated"},
        "glass": {"fill_percent": 92, "source": "simulated"},
        "hazardous": {"fill_percent": 15, "source": "simulated"},
        "biodegradable": {"fill_percent": 60, "source": "simulated"},
    }

    for item in ["plastic bottle", "battery", "fruit peel"]:
        print("=" * 60)
        print(item)
        print(
            generate_recycling_advice(
                waste_type=item,
                confidence=0.87,
                bin_status=test_bins,
            )
        )