"""rewabuddy_advisor.py - ReWA AI Buddy Recycling & Reuse Advisor.

Multi-provider AI assistant with automatic fallback:
1. Google Gemini (google-genai SDK, gemini-2.5-flash / gemini-2.0-flash / gemini-1.5-flash)
2. Groq (llama-3.3-70b-versatile / llama-3.1-8b-instant)
3. OpenRouter (meta-llama / gemini free tiers)
4. Offline Rule-Based Advisor (Python knowledge base; zero cost, 100% available)

Features:
- Reads API keys from environment variables AND Streamlit secrets (.streamlit/secrets.toml)
- Handles Unicode / Windows encoding safely via Python logging
- Deterministic material classification & bin routing decided by Python logic
- AI explains facts without hallucinating bin levels or unsafe reuse
- Safe handling rules for hazardous waste (no DIY reuse for batteries/chemicals)
"""

import logging
import os
import time
import requests

from recycling_logic import decide_bin

logger = logging.getLogger(__name__)

# ============================================================
# SMART RECYCLING ASSISTANT — CONFIGURATION
# ============================================================

# Valid, modern Google Gemini models supported by google-genai
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Supported models on Groq
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "openai/gpt-oss-120b",
]

# Supported free / standard models on OpenRouter
OPENROUTER_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "openrouter/free",
]

REQUEST_TIMEOUT = 30
MIN_CONFIDENCE_FOR_AUTOMATIC_SORTING = 0.55
BIN_FULL_THRESHOLD = 90


# ============================================================
# SECRET / API KEY RETRIEVAL (ENV + STREAMLIT SECRETS)
# ============================================================

def get_api_key(primary_name, alternates=None):
    """
    Retrieve an API key from environment variables or Streamlit secrets.
    Checks:
    1. os.getenv(primary_name) & alternates
    2. st.secrets[primary_name] & alternates
    3. st.secrets nested sections (e.g. st.secrets["gemini"]["api_key"])
    """
    # 1. Environment variables
    val = os.getenv(primary_name)
    if val and val.strip():
        return val.strip()

    if alternates:
        for alt in alternates:
            val = os.getenv(alt)
            if val and val.strip():
                return val.strip()

    # 2. Streamlit secrets
    try:
        import streamlit as st

        # Direct top-level check
        if primary_name in st.secrets:
            return str(st.secrets[primary_name]).strip()
        if alternates:
            for alt in alternates:
                if alt in st.secrets:
                    return str(st.secrets[alt]).strip()

        # Section-level check
        for _, sec_val in st.secrets.items():
            if isinstance(sec_val, dict):
                for k in [primary_name, primary_name.lower(), "api_key", "key", "token"]:
                    if k in sec_val and sec_val[k]:
                        return str(sec_val[k]).strip()
                if alternates:
                    for alt in alternates:
                        if alt in sec_val and sec_val[alt]:
                            return str(sec_val[alt]).strip()
                        if alt.lower() in sec_val and sec_val[alt.lower()]:
                            return str(sec_val[alt.lower()]).strip()
    except Exception:
        pass

    return None


# ============================================================
# MATERIAL CLASSIFICATION (decided by Python, not by the AI)
# ============================================================

HAZARDOUS_KEYWORDS = [
    "battery", "e-waste", "electronic", "chemical", "bulb",
    "medicine", "paint", "aerosol", "syringe", "tube light",
    "cfl", "pesticide", "thermometer", "toner", "biohazard",
]

BIODEGRADABLE_KEYWORDS = [
    "food", "leaf", "leaves", "fruit", "vegetable", "organic",
    "peel", "garden", "eggshell", "tea leaves", "coffee grounds",
    "flower", "grass",
]

PACKAGING_WORDS = ["plastic", "wrapper", "packet", "packaging", "bottle", "can", "container"]


def classify_material(waste_type):
    """Return 'hazardous', 'biodegradable', or 'recyclable and reusable'."""
    label = str(waste_type or "").strip().lower()

    if any(word in label for word in HAZARDOUS_KEYWORDS):
        return "hazardous"

    if any(word in label for word in BIODEGRADABLE_KEYWORDS):
        if not any(word in label for word in PACKAGING_WORDS):
            return "biodegradable"

    return "recyclable and reusable"


# ============================================================
# BIN DECISION FOR HAZARDOUS / BIODEGRADABLE ITEMS
# ============================================================

SPECIAL_BINS = {
    "hazardous": "Hazardous",
    "biodegradable": "Biodegradable",
}


def decide_special_bin(material_class, confidence, bin_status):
    """Python-only bin decision for hazardous and biodegradable items."""
    bin_name = SPECIAL_BINS.get(material_class, "Hazardous")

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
            "separate from ordinary waste and dry recyclables."
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
# REUSE KNOWLEDGE BASE
# ============================================================

ITEM_REUSE = {
    "plastic bottle": [
        "Cut the bottle and use the bottom half as a pen stand or a small planter.",
        "Poke holes in the cap and use it as a gentle watering can for plants.",
        "Use a thoroughly cleaned bottle to store dry items like pulses, seeds, or stationery.",
    ],
    "plastic bag": [
        "Reuse it as a bin liner for small bedroom or bathroom dustbins.",
        "Keep it folded in your backpack for wet umbrellas or grocery runs.",
        "Use it to protect books or electronics from unexpected rain.",
    ],
    "plastic wrapper": [
        "Cut clean wrappers into strips for craft, collage, or weaving projects.",
        "Use clean bubble wrap or plastic film as cushioning when packing parcels.",
        "Collect clean wrappers for municipal dry waste co-processing / DWCC drop-off.",
    ],
    "plastic container": [
        "Wash well and use to organise screws, buttons, craft supplies, or cables.",
        "Use as a drip tray under plant pots to protect window sills.",
        "Use as drawer dividers to organise loose office stationery.",
    ],
    "cardboard box": [
        "Flatten and store for moving, shipping, or storage.",
        "Cut into drawer dividers or a DIY desk organiser.",
        "Use as a protective surface when painting or working on craft projects.",
    ],
    "paper sheet": [
        "Use the reverse blank side for rough math, notes, or drafts.",
        "Cut clean unused sections into pocket notepads or bookmarks.",
        "Shred clean unbleached paper for compost brown matter or parcel packing.",
    ],
    "newspaper": [
        "Use to wrap fragile glass items, gifts, or parcel contents.",
        "Line kitchen drawers, shoe racks, or pet trays.",
        "Crumple with a drop of vinegar to clean glass and mirrors streak-free.",
    ],
    "metal can": [
        "Wash thoroughly, sand down or tape sharp rims, and use as a pencil holder.",
        "Drill small drainage holes in the bottom and paint as a rustic planter.",
        "Organise paintbrushes, kitchen utensils, or workshop tools.",
    ],
    "glass bottle": [
        "Wash and sterilise to use as a fridge water bottle or flower vase.",
        "Fill with fairy lights or sand to make an ambient desk ornament.",
        "Reuse for homemade cold-brew, iced tea, or decorative storage.",
    ],
    "glass jar": [
        "Wash and reuse to store spices, homemade jam, overnight oats, or pulses.",
        "Use as a chic desk organiser for pens, scissors, and paperclips.",
        "Place a small tea-light candle inside for an ambient lantern.",
    ],
}

CATEGORY_REUSE = [
    (["plastic"], ITEM_REUSE["plastic container"]),
    (["cardboard", "carton", "box"], ITEM_REUSE["cardboard box"]),
    (["paper"], ITEM_REUSE["paper sheet"]),
    (["metal", "tin", "can", "aluminium", "aluminum"], ITEM_REUSE["metal can"]),
    (["glass", "jar"], ITEM_REUSE["glass jar"]),
]

GENERIC_REUSE = [
    "Check whether the item can be cleaned and repurposed for storage or containment.",
    "Donate it if it remains intact and in usable condition.",
    "Repurpose it for a DIY craft or college project before final disposal.",
]

BIODEGRADABLE_REUSE = {
    "food waste": [
        "Add fruit and vegetable trimmings to a home compost bucket or community compost pit.",
        "Use a small kitchen bokashi bin or terracotta composter if living in an apartment.",
        "Hand over to municipal wet waste collectors separately; never mix with plastics.",
    ],
    "fruit peel": [
        "Bury citrus or banana peels under plant soil to add potassium and organic nutrients.",
        "Ferment citrus peels with jaggery and water to produce a natural eco-enzyme cleaner.",
        "Add peels to a compost bucket to enrich organic compost matter.",
    ],
    "leaves": [
        "Use dry leaves as mulch around potted plants to retain soil moisture.",
        "Add dry leaves as 'brown' carbon matter to balance nitrogen-rich kitchen waste in compost.",
        "Use clean dried leaves for craft, art projects, or garden bed bedding.",
    ],
}

BIODEGRADABLE_DEFAULT = [
    "Compost in a home compost bin, garden compost pit, or community wet waste facility.",
    "Bury small organic scraps in plant soil so they decompose naturally.",
    "Keep strictly separated from plastic, metal, and dry recyclables.",
]

HAZARDOUS_GUIDE = {
    "battery": {
        "reuse_line": (
            "⚠️ Do not reuse or open spent batteries. Switch to rechargeable "
            "batteries to significantly reduce hazardous chemical waste."
        ),
        "handling": [
            "Never throw batteries in ordinary dustbins or incinerate them.",
            "Cover terminal ends of lithium or button cells with insulating tape to prevent short circuits.",
            "If a battery is swollen or leaking, handle with gloves and deposit at a designated E-waste drop-off.",
        ],
    },
    "light bulb": {
        "reuse_line": (
            "⚠️ Do not attempt DIY crafts with damaged or expired bulbs. "
            "Intact working bulbs may be repurposed into other fixtures."
        ),
        "handling": [
            "CFLs and fluorescent tubes contain mercury vapour — never discard in general trash.",
            "Wrap unbroken bulbs in paper or original packaging before transporting to hazardous disposal.",
            "If broken, ventilate the room for 15 minutes and pick up fragments with stiff paper; do not vacuum.",
        ],
    },
    "medicine": {
        "reuse_line": (
            "⚠️ Never reuse or share expired or leftover medicines. "
            "Return them via pharmacy take-back channels."
        ),
        "handling": [
            "Return unneeded medications to authorized pharmacy take-back collection boxes.",
            "Do not flush medications down the sink or toilet unless explicitly instructed on packaging.",
            "Keep in original child-resistant containers until safely deposited at hazardous drop-offs.",
        ],
    },
    "default": {
        "reuse_line": (
            "⚠️ This is hazardous waste — do not attempt DIY reuse. "
            "Deposit at an authorized hazardous or E-waste collection point."
        ),
        "handling": [
            "Keep strictly isolated from household dry recyclables and wet compost.",
            "Use authorized domestic hazardous waste collection centres or e-waste kiosks.",
            "Keep in original containers away from heat sources, children, and pets.",
        ],
    },
}


def get_reuse_info(waste_type, material_class):
    """Return reuse suggestions and safe handling instructions."""
    label = str(waste_type or "").strip().lower()

    if material_class == "hazardous":
        guide = HAZARDOUS_GUIDE["default"]
        for key in ("battery", "light bulb", "medicine"):
            if key in label or key.split()[-1] in label:
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
# PROVIDER 1 — GOOGLE GEMINI
# ============================================================

def try_gemini(prompt):
    api_key = get_api_key(
        "GEMINI_API_KEY",
        alternates=["GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"]
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize Google GenAI client: {exc}")

    last_error = None
    for model in GEMINI_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                if not response or not response.text:
                    raise RuntimeError("Gemini returned an empty response.")
                return response.text.strip(), f"Google Gemini ({model})"
            except Exception as error:
                last_error = error
                err_str = str(error)
                # If model is not found, immediately try next model
                if "404" in err_str or "not found" in err_str.lower():
                    break
                temporary_error = any(code in err_str for code in ["429", "500", "502", "503", "504"])
                if temporary_error and attempt == 0:
                    time.sleep(2)
                    continue
                break

    raise RuntimeError(f"Gemini failed: {last_error}")


# ============================================================
# SHARED OPENAI-COMPATIBLE API CALL (GROQ & OPENROUTER)
# ============================================================

def call_openai_style(url, api_key, model, prompt, label):
    if not api_key:
        raise RuntimeError(f"{label} API key is not configured.")
    if not model:
        raise RuntimeError(f"{label} model is not specified.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in url:
        headers["HTTP-Referer"] = "https://rewa-ai.streamlit.app"
        headers["X-Title"] = "ReWA.AI"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ReWA AI Buddy, a practical, encouraging recycling and "
                    "reuse advisor for ReWA.AI. "
                    "Use strictly the verified facts supplied in the prompt. "
                    "Never invent measurements, fill levels, or sensor readings. "
                    "Explain uncertainty clearly and provide safe, actionable advice."
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
        raise RuntimeError(f"{label} returned an unexpected response format.") from error

    if isinstance(answer, list):
        answer = "\n".join(part.get("text", "") for part in answer if isinstance(part, dict))

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"{label} returned an empty response.")

    return answer.strip(), f"{label} ({model})"


# ============================================================
# PROVIDER 2 — GROQ
# ============================================================

def try_groq(prompt):
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    last_error = None
    for model in GROQ_MODELS:
        try:
            return call_openai_style(
                url="https://api.groq.com/openai/v1/chat/completions",
                api_key=api_key,
                model=model,
                prompt=prompt,
                label="Groq",
            )
        except Exception as error:
            last_error = error
            continue

    raise RuntimeError(f"Groq failed: {last_error}")


# ============================================================
# PROVIDER 3 — OPENROUTER
# ============================================================

def try_openrouter(prompt):
    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    last_error = None
    for model in OPENROUTER_MODELS:
        try:
            return call_openai_style(
                url="https://openrouter.ai/api/v1/chat/completions",
                api_key=api_key,
                model=model,
                prompt=prompt,
                label="OpenRouter",
            )
        except Exception as error:
            last_error = error
            continue

    raise RuntimeError(f"OpenRouter failed: {last_error}")


# ============================================================
# AI PROVIDER FALLBACK
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
            logger.info("ReWA AI Buddy advice generated via %s", source)
            return answer, source
        except Exception as error:
            message = str(error)
            logger.info("%s failed: %s", provider.__name__, message[:200])
            errors.append(f"{provider.__name__}: {message[:200]}")

    raise RuntimeError("All configured AI providers failed. " + " | ".join(errors))


# ============================================================
# PYTHON-ONLY FALLBACK (OFFLINE INTELLIGENCE)
# ============================================================

def offline_advice(waste_type, decision, material_class="recyclable and reusable"):
    reuse_info = get_reuse_info(waste_type, material_class)

    lines = [
        "## ♻️ Recycling Advice",
        f"Detected item: **{waste_type}** (material category: *{material_class}*).",
    ]

    if decision.get("needs_verification", True):
        lines.append(
            "⚠️ *Please verify the physical material markings before final bin disposal.*"
        )

    if decision.get("note"):
        lines.append(decision["note"])

    lines.append("## 🗑️ Recommended Bin")

    bin_name = decision.get("bin")
    if bin_name is None:
        lines.append("No matching canonical bin is available in this station. Consult municipal guidelines.")
    else:
        fill = decision.get("fill_percent")
        if fill is None:
            lines.append(f"The suggested bin is **{bin_name}**.")
        else:
            lines.append(f"The suggested bin is **{bin_name}** (simulated fill: {fill}%).")

        if decision.get("is_full"):
            lines.append("⚠️ *This bin is nearly full (≥90%). Consider an alternative collection point or DWCC.*")

    lines.append("## 🔁 Reuse Ideas")

    if material_class == "hazardous":
        lines.append(reuse_info["reuse_line"])
    elif material_class == "biodegradable":
        lines.append(
            "This item is biodegradable. Compost it instead of discarding:\n"
            + "\n".join(f"- {idea}" for idea in reuse_info["ideas"])
        )
    else:
        lines.append("\n".join(f"- {idea}" for idea in reuse_info["ideas"]))

    lines.append("## 💡 Smart Suggestion")

    if material_class == "hazardous":
        tips = reuse_info["handling"]
    else:
        tips = [
            "Check the item's material markings or resin identification code.",
            "Keep recyclable materials clean, dry, and emptied before binning.",
            "Follow your local ward (BBMP) collection schedules.",
        ]

    lines.append("\n".join(f"- {tip}" for tip in tips))

    return "\n\n".join(lines)


# ============================================================
# MAIN RECYCLING ADVISOR INTERFACE
# ============================================================

def generate_recycling_advice(
    waste_type,
    confidence=None,
    bin_status=None,
    score=None,
    bins=None,
):
    """
    Generate recycling AND reuse advice using AI cloud providers
    with an automatic Python-only fallback.
    Never throws unhandled exceptions to the caller.
    """
    try:
        # Support legacy argument aliases
        if score is not None:
            confidence = score
        if bins is not None:
            bin_status = bins
        if confidence is None:
            confidence = 0.0
        if bin_status is None:
            bin_status = {}

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))

        # 1. Deterministic Python material classification
        material_class = classify_material(waste_type)

        # 2. Deterministic Python bin routing
        if material_class in SPECIAL_BINS:
            decision = decide_special_bin(material_class, confidence, bin_status)
        else:
            decision = decide_bin(waste_type, confidence, bin_status)

        # 3. Deterministic reuse suggestions and safety guidelines
        reuse_info = get_reuse_info(waste_type, material_class)

        if material_class == "hazardous":
            reuse_block = (
                f"- Safety Notice: {reuse_info['reuse_line']}\n"
                "- Safe handling tips:\n"
                + "\n".join(f"  * {tip}" for tip in reuse_info["handling"])
            )
        else:
            reuse_block = "\n".join(f"- {idea}" for idea in reuse_info["ideas"])

        prompt = f"""
You are ReWA AI Buddy, a practical, encouraging recycling and reuse advisor for ReWA.AI.

Explain the following verified facts in clear, student-friendly English.

VERIFIED SYSTEM FACTS:
- Detected item label: {waste_type}
- Material category: {material_class}
- Vision similarity score: {confidence:.3f} (Note: This is an image similarity score, not a guaranteed probability)
- Suggested bin: {decision.get("bin")}
- Bin fill level: {decision.get("fill_percent")} (simulated in Digital Twin, not physical hardware sensors)
- Bin is nearly full: {decision.get("is_full")}
- Material verification needed: {decision.get("needs_verification")}
- Additional note: {decision.get("note")}

REUSE & SAFETY GUIDELINES:
{reuse_block}

INSTRUCTIONS:
1. Do not claim the vision classification is 100% infallible.
2. If verification is needed, advise checking material markings or labels.
3. Strictly adhere to the verified system facts above. Never invent fake sensor numbers.
4. If a value is unavailable, clearly mention that it is unavailable.
5. If the suggested bin is nearly full, recommend finding another collection point or reporting it.
6. Mention that municipal recycling guidelines can vary by ward or locality.
7. Under Reuse Ideas, provide safe, creative, doable reuse or composting suggestions based on the ideas above.
8. For hazardous items (batteries, bulbs, medicines, e-waste, chemicals), NEVER suggest DIY craft reuse. Provide safe handling and authorized disposal guidance.

Use exactly these four Markdown headings:

## ♻️ Recycling Advice

## 🗑️ Recommended Bin

## 🔁 Reuse Ideas

## 💡 Smart Suggestion

Under Reuse Ideas, provide 3 concise bullet points (or safe handling for hazardous waste).
Under Smart Suggestion, provide 3 practical tips.
"""

        try:
            answer, source = ask_ai(prompt)
            return f"{answer}\n\n*Answered by: {source}*"
        except Exception as ai_err:
            logger.info("AI cloud providers unavailable (%s); using offline intelligence.", ai_err)
            return (
                offline_advice(waste_type, decision, material_class)
                + "\n\n*Answered by: ReWA AI Buddy (Built-in Knowledge Base)*"
            )

    except Exception as fatal_error:
        logger.error("Error in generate_recycling_advice: %s", fatal_error)
        clean_name = str(waste_type or "Item").title()
        return (
            f"## ♻️ Recycling Advice\n\n"
            f"Detected item: **{clean_name}**.\n\n"
            f"## 🗑️ Recommended Bin\n\n"
            f"Please check municipal segregation rules or verify the item's material markings.\n\n"
            f"## 🔁 Reuse Ideas\n\n"
            f"- Clean and reuse for storage or containers if intact.\n"
            f"- Donate if still in working or usable condition.\n\n"
            f"## 💡 Smart Suggestion\n\n"
            f"- Rinse and dry recyclables before placing them in the bin.\n"
            f"- Check local ward (BBMP) collection schedules.\n\n"
            f"*Answered by: ReWA AI Buddy (Safety Fallback)*"
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
        print(generate_recycling_advice(waste_type=item, confidence=0.87, bin_status=test_bins))
