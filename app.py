import base64
import hashlib
import io
import os
import sqlite3
import uuid
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline

import data_base
import rewards as rewards_module
from data_base import create_data_base, get_events, get_bin_totals
from digital_twin import create_bins, record_item, empty_bin, get_bin_for_item
from rewabuddy_advisor import generate_recycling_advice
from rewards import (
    DAILY_GOAL,
    POINTS_PER_EVENT,
    award_event,
    get_daily_history,
    get_points_by_category,
    get_summary,
    init_rewards_db,
)

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="ReWA.AI",
    page_icon="♻️",
    layout="wide"
)
create_data_base()
init_rewards_db()

# ---------- LUXURY ECO PALETTE ----------
EMERALD = "#0B2E22"     # deepest green: buddy banner, button gradient
FOREST = "#174B32"      # primary: buttons, headings
LEAF = "#2B6E4B"        # mid green
GOLD = "#C9A24B"        # accent: borders, dividers, highlights
CHAMPAGNE = "#F3E6C4"   # soft gold: badges, subtitle
MINT = "#A8E6C0"        # success
IVORY = "#FBF8F1"       # page background
PALE_LEAF = "#E8F3EA"   # cards, subtle panels
INK = "#14221A"         # body text

# One consistent colour per bin, used by every chart and bin card
BIN_COLORS = {
    "Plastic": "#2F80ED",
    "Paper": "#E0A526",
    "Metal": "#7A8794",
    "Glass": "#1FA39A",
    "Hazardous": "#D64545",
    "Biodegradable": "#4CAF50",
}
BIN_ICONS = {
    "Plastic": "♻️",
    "Paper": "📄",
    "Metal": "🔩",
    "Glass": "🍾",
    "Hazardous": "☣️",
    "Biodegradable": "🌿",
}

# ---------- IMAGE INPUT SETTINGS ----------
# The AI model runs on your computer, so a very large batch can be slow.
# Raise or lower this number to suit your hardware.
MAX_IMAGES = 10

# ---------- WASTE LABELS ----------
# Labels the AI model chooses from. Hazardous items have several labels so
# they are no longer swallowed by "other waste".
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

# Any label containing one of these words is treated as hazardous.
# (Labels are compared in lowercase with underscores instead of spaces.)
HAZARDOUS_KEYWORDS = [
    "battery",
    "light_bulb",
    "medicine",
    "syringe",
    "needle",
    "sharps",
    "medical",
    "biohazard",
    "chemical",
    "aerosol",
    "paint",
    "pesticide",
    "poison",
    "hazardous",
]

# Subset of the above that needs the special "sharps / medical" advice.
SHARPS_KEYWORDS = ["syringe", "needle", "sharps", "medical", "biohazard"]


def normalize_label(label):
    return label.lower().replace(" ", "_")


def is_hazardous_label(label):
    label = normalize_label(label)
    return any(word in label for word in HAZARDOUS_KEYWORDS)


def resolve_bin(label):
    """Hazardous items go to the Hazardous bin; otherwise use the original mapping."""
    if is_hazardous_label(label):
        return "Hazardous"
    return get_bin_for_item(label)


# ---------- BANNER IMAGE (OPTIONAL) ----------
# The banner is drawn with HTML/CSS/SVG, so it stays sharp on every screen.
# If you would rather show your own exported banner picture instead, save it
# next to this file as rewa_banner.jpg / .jpeg / .png and set this to True.
USE_BANNER_IMAGE = False
BANNER_FILE_NAMES = ["rewa_banner.jpg", "rewa_banner.jpeg", "rewa_banner.png"]


def load_banner_base64():
    """Return (mime_type, base64_string) for the banner image, or (None, None)."""
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    for name in BANNER_FILE_NAMES:
        path = base_dir / name
        if path.exists():
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            return mime, base64.b64encode(path.read_bytes()).decode()
    return None, None


# ---------- BANNER GRAPHICS (inline SVG) ----------
# Recycling arrows + leaf mark that sits above the wordmark.
LOGO_MARK_SVG = (
    '<svg class="rewa-mark" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<linearGradient id="mkA" gradientUnits="userSpaceOnUse" x1="14" y1="20" x2="100" y2="108">'
    '<stop offset="0%" stop-color="#D8F8B6"/>'
    '<stop offset="100%" stop-color="#8FDB7A"/>'
    '</linearGradient>'
    '<linearGradient id="mkB" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0%" stop-color="#EAFBC8"/>'
    '<stop offset="100%" stop-color="#9BE08A"/>'
    '</linearGradient>'
    '</defs>'
    '<g fill="url(#mkA)" stroke="url(#mkA)" stroke-linejoin="round" stroke-linecap="round">'
    '<path d="M54 22 L30 64" stroke-width="12" fill="none"/>'
    '<polygon points="19,83 42,71 18,57" stroke-width="3"/>'
    '<g transform="rotate(-120 60 66)">'
    '<path d="M54 22 L30 64" stroke-width="12" fill="none"/>'
    '<polygon points="19,83 42,71 18,57" stroke-width="3"/>'
    '</g>'
    '</g>'
    '<path d="M94 78 Q104 42.6 68 33 Q58.6 68.2 94 78 Z" fill="url(#mkB)"/>'
    '<path d="M90 73 L74 41" stroke="#1F6F3A" stroke-width="2.4" '
    'fill="none" stroke-linecap="round" opacity="0.8"/>'
    '</svg>'
)

# Tiny leaf that sits on the "A" of the wordmark.
WORD_LEAF_SVG = (
    '<svg class="w-leaf" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M3 21 C3 9 11 3 22 3 C22 14 16 21 3 21 Z" fill="#1F7A47"/>'
    '<path d="M3 21 L15 9" stroke="#D6F7B8" stroke-width="1.5" fill="none" '
    'stroke-linecap="round"/>'
    '</svg>'
)

# Small leaf used in the divider line.
DIVIDER_LEAF_SVG = (
    '<svg class="rule-leaf" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M3 21 C3 9 11 3 22 3 C22 14 16 21 3 21 Z" fill="#E3F8D0"/>'
    '<path d="M3 21 L15 9" stroke="#1F6F49" stroke-width="1.4" fill="none" '
    'stroke-linecap="round"/>'
    '</svg>'
)

# Decorative globe + leaf swoosh in the bottom-left corner.
GLOBE_SVG = (
    '<svg class="globe-leaf" viewBox="0 0 380 260" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<linearGradient id="gl" x1="0" y1="1" x2="1" y2="0">'
    '<stop offset="0%" stop-color="#4FA55E"/>'
    '<stop offset="60%" stop-color="#A9DE94"/>'
    '<stop offset="100%" stop-color="#E4F8C8"/>'
    '</linearGradient>'
    '</defs>'
    '<path d="M0 196 C110 236 250 206 296 112 C276 204 160 258 0 250 Z" fill="url(#gl)" opacity="0.95"/>'
    '<path d="M0 222 C130 240 250 205 300 114" stroke="#D9F7B5" stroke-width="2" '
    'fill="none" opacity="0.7"/>'
    '</svg>'
)

# Four outlined icon tiles on the right-hand side.
_ICON_ATTR = (
    'viewBox="0 0 24 24" fill="none" stroke="#EAF9DF" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg"'
)
TILES_HTML = (
    '<div class="tiles">'
    f'<div class="tile"><svg {_ICON_ATTR}>'
    '<path d="M10 2h4v3l1.5 2.5V21a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1V7.5L10 5z"/>'
    '<path d="M8.5 11h7M8.5 16h7"/></svg></div>'
    f'<div class="tile"><svg {_ICON_ATTR}>'
    '<path d="M6 5c0-1.1 2.7-2 6-2s6 .9 6 2v14c0 1.1-2.7 2-6 2s-6-.9-6-2z"/>'
    '<path d="M6 5c0 1.1 2.7 2 6 2s6-.9 6-2M6 9c0 1.1 2.7 2 6 2s6-.9 6-2"/></svg></div>'
    f'<div class="tile"><svg {_ICON_ATTR}>'
    '<path d="M7 19H4.815a1.83 1.83 0 0 1-1.57-.881 1.785 1.785 0 0 1-.004-1.784L7.196 9.5"/>'
    '<path d="M11 19h8.203a1.83 1.83 0 0 0 1.556-.89 1.784 1.784 0 0 0 0-1.775l-1.226-2.12"/>'
    '<path d="m14 16-3 3 3 3"/>'
    '<path d="M8.293 13.596 7.196 9.5 3.1 10.598"/>'
    '<path d="m9.344 5.811 1.093-1.892A1.83 1.83 0 0 1 11.985 3a1.784 1.784 0 0 1 1.546.888l3.943 6.843"/>'
    '<path d="m13.378 9.633 4.096 1.098 1.097-4.096"/></svg></div>'
    f'<div class="tile"><svg {_ICON_ATTR}>'
    '<path d="M4 20C4 10 10 4 20 4c0 10-6 16-16 16z"/><path d="M4 20L14 10"/></svg></div>'
    '</div>'
)

# ---------- SESSION STATE ----------
# List of analysed images. Each entry is a dict:
#   uid, name, image (PIL), results (top-3 predictions), best_label, best_score
if "batch" not in st.session_state:
    st.session_state.batch = []

# uids of items that were already recorded in a recycling bin
if "recorded_uids" not in st.session_state:
    st.session_state.recorded_uids = set()

# Which bin each recorded item went into: {uid: bin_name}
if "recorded_bins" not in st.session_state:
    st.session_state.recorded_bins = {}

# Photos taken with the live camera, waiting to be classified:
#   [{"id": md5, "name": "Camera photo 1", "data": bytes}, ...]
if "camera_shots" not in st.session_state:
    st.session_state.camera_shots = []

# Changing these numbers gives the uploader / camera a fresh, empty widget
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0

if "digital_twin_bins" not in st.session_state:
    st.session_state.digital_twin_bins = create_bins()

    saved_totals = get_bin_totals()

    for material, total in saved_totals.items():
        if material in st.session_state.digital_twin_bins:
            bin_data = st.session_state.digital_twin_bins[material]
            bin_data["item_count"] = total
            bin_data["capacity_percent"] = min(total * 5, 100)

if "flash" not in st.session_state:
    st.session_state.flash = None

# True while the "Are you sure?" step of the Eco Points reset is showing
if "confirm_reset_rewards" not in st.session_state:
    st.session_state.confirm_reset_rewards = False

# ReWA AI Buddy answers, one per image: {uid: advice_text}
if "buddy_advice" not in st.session_state or not isinstance(
    st.session_state.buddy_advice, dict
):
    st.session_state.buddy_advice = {}

# ---------- LUXURY THEME ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Playfair+Display:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

.stApp {{
    background:
        radial-gradient(circle at 8% 0%, rgba(201, 162, 75, 0.10), transparent 40%),
        radial-gradient(circle at 100% 30%, rgba(23, 75, 50, 0.06), transparent 45%),
        {IVORY};
    color: {INK};
    font-family: 'Inter', sans-serif;
}}

/* Make Streamlit's top bar transparent so it never cuts the banner */
header[data-testid="stHeader"] {{
    background: transparent !important;
}}

.block-container {{
    padding: 1.8rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}}

/* Readable body text */
.stApp p,
.stApp label,
.stApp li,
.stMarkdown,
.stCaption {{
    color: {INK};
    font-family: 'Inter', sans-serif;
}}

/* Headings: elegant serif */
h1, h2, h3, h4 {{
    font-family: 'Playfair Display', serif !important;
    color: {FOREST} !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px;
}}

h2 {{
    padding-bottom: 6px;
}}

/* Gold hairline dividers */
hr {{
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, {GOLD}, transparent) !important;
    margin: 2rem 0 !important;
}}

/* =====================================================================
   REWA.AI HERO BANNER  (faded green, compact, slightly wider than content)
   ===================================================================== */
.logo-banner {{
    position: relative;
    overflow: hidden;
    isolation: isolate;
    min-height: 268px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px 120px;
    margin: 0 -1.5rem 26px -1.5rem;
    border-radius: 26px;
    border: 1px solid rgba(201, 162, 75, 0.55);
    background:
        radial-gradient(ellipse 55% 90% at 96% 0%, rgba(214, 240, 190, 0.50), transparent 62%),
        radial-gradient(ellipse 45% 70% at 50% 115%, rgba(190, 235, 170, 0.22), transparent 70%),
        linear-gradient(115deg, #1D6E47 0%, #2F8554 48%, #86C28F 100%);
    box-shadow: 0 16px 34px rgba(31, 111, 73, 0.24),
                inset 0 1px 0 rgba(255, 255, 255, 0.18);
}}

/* thin gold hairline along the top edge */
.logo-banner::before {{
    content: "";
    position: absolute;
    top: 0; left: 60px; right: 60px;
    height: 2px;
    z-index: 3;
    background: linear-gradient(90deg, transparent, {GOLD}, transparent);
}}

/* ---------- centre column ---------- */
.banner-center {{
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}}

.banner-center .rewa-mark {{
    width: clamp(52px, 5.5vw, 70px);
    height: auto;
    margin-bottom: 4px;
    filter: drop-shadow(0 6px 10px rgba(0, 0, 0, 0.25));
}}

/* Wordmark: clean modern sans-serif */
.stApp .banner-center .wordmark {{
    font-family: 'Manrope', 'Inter', sans-serif;
    font-weight: 700;
    font-size: clamp(2.4rem, 5.2vw, 4rem);
    line-height: 1.05;
    letter-spacing: -1px;
    margin: 0;
    color: #FFFFFF !important;
    text-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
    white-space: nowrap;
}}

.stApp .banner-center .wordmark .w-white {{
    color: #FFFFFF !important;
}}

.stApp .banner-center .wordmark .w-green {{
    position: relative;
    color: #B8F27E !important;
    background: linear-gradient(180deg, #D2F9A0, #A4E86A);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.banner-center .a-wrap {{
    position: relative;
    display: inline-block;
    line-height: 1;
    background: linear-gradient(180deg, #D2F9A0, #A4E86A);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.banner-center .w-leaf {{
    position: absolute;
    left: 0.36em;
    bottom: 0.16em;
    width: 0.30em;
    height: 0.30em;
    filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.25));
    -webkit-text-fill-color: initial;
}}

.stApp .banner-center .full-name {{
    font-family: 'Manrope', 'Inter', sans-serif;
    font-weight: 500;
    font-size: clamp(0.9rem, 1.6vw, 1.25rem);
    letter-spacing: 0.3px;
    margin: 8px 0 0 0;
    color: #F0FAEA !important;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.22);
}}

/* divider: line - leaf - line */
.banner-center .rule {{
    display: flex;
    align-items: center;
    gap: 12px;
    width: min(320px, 70%);
    margin: 14px 0 12px 0;
}}

.banner-center .rule span.line {{
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232, 250, 220, 0.85));
}}

.banner-center .rule span.line.r {{
    background: linear-gradient(270deg, transparent, rgba(232, 250, 220, 0.85));
}}

.banner-center .rule-leaf {{
    width: 16px;
    height: 16px;
    flex-shrink: 0;
}}

.stApp .banner-center .tagline {{
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: clamp(0.62rem, 0.95vw, 0.78rem);
    text-transform: uppercase;
    letter-spacing: 0.42em;
    padding-left: 0.42em;
    margin: 0;
    color: #EAF8DF !important;
}}

.stApp .banner-center .tagline .dot {{
    margin: 0 0.9em;
    color: #D2F9A0 !important;
}}

/* ---------- globe (bottom-left) ----------
   The globe and the leaf swoosh live in ONE wrapper, so they always line up. */
.globe-wrap {{
    position: absolute;
    z-index: 1;
    left: 0;
    bottom: 0;
    width: 300px;
    height: 205px;
    pointer-events: none;
}}

.globe {{
    position: absolute;
    left: -63px;
    top: 32px;
    width: 332px;
    height: 332px;
    border-radius: 50%;
    background:
        radial-gradient(circle at 62% 34%, rgba(200, 240, 180, 0.45) 0 8%, transparent 9%),
        radial-gradient(circle at 48% 46%, rgba(30, 110, 60, 0.45) 0 14%, transparent 15%),
        radial-gradient(circle at 72% 58%, rgba(30, 110, 60, 0.40) 0 10%, transparent 11%),
        radial-gradient(circle at 70% 28%, #7CC27C 0%, #3E9157 42%, #226B43 72%, #165A38 100%);
    box-shadow: 0 0 50px rgba(200, 240, 170, 0.35),
                inset -20px -16px 46px rgba(0, 0, 0, 0.28);
    opacity: 0.95;
}}

.globe-leaf {{
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    filter: drop-shadow(0 5px 8px rgba(0, 0, 0, 0.22));
}}

/* ---------- icon tiles (right) ---------- */
.tiles {{
    position: absolute;
    z-index: 2;
    right: 34px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    gap: 10px;
}}

.tile {{
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    border: 1px solid rgba(236, 251, 226, 0.55);
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(2px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14);
}}

.tile svg {{
    width: 22px;
    height: 22px;
}}

/* Optional: user-supplied banner picture */
.logo-banner.image-mode {{
    padding: 0;
    min-height: 0;
    display: block;
}}

.logo-banner img.logo-img {{
    display: block;
    width: 100%;
    height: auto;
}}

/* ---------- responsive ---------- */
@media (max-width: 1100px) {{
    .logo-banner {{ padding: 24px 90px; }}
    .globe-wrap {{ left: -40px; }}
}}

@media (max-width: 800px) {{
    .block-container {{ padding: 1.6rem 1rem 2rem 1rem; }}
    .logo-banner {{
        padding: 26px 16px;
        min-height: 232px;
        margin: 0 -0.4rem 20px -0.4rem;
        border-radius: 22px;
    }}
    .tiles {{ display: none; }}
    .globe-wrap {{ opacity: 0.5; }}
    .stApp .banner-center .tagline {{ letter-spacing: 0.25em; padding-left: 0.25em; }}
}}

/* ---------- BORDERED PANELS (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(201, 162, 75, 0.35) !important;
    border-radius: 20px !important;
    padding: 8px 10px;
    box-shadow: 0 10px 28px rgba(23, 75, 50, 0.08);
}}

/* AI result card */
.result {{
    background: linear-gradient(135deg, {PALE_LEAF}, #F7FBF4);
    border-radius: 16px;
    padding: 18px 22px;
    border-left: 6px solid {GOLD};
    margin: 10px 0 14px 0;
}}

.result h3 {{
    color: {FOREST} !important;
    margin: 0 0 4px 0 !important;
}}

.result p {{
    color: {INK} !important;
    margin: 0 !important;
}}

/* ReWA AI Buddy banner */
.buddy-head {{
    background: linear-gradient(120deg, {EMERALD}, {FOREST});
    border: 1px solid rgba(201, 162, 75, 0.6);
    border-radius: 16px;
    padding: 16px 22px;
    margin: 18px 0 12px 0;
}}

.stApp .buddy-head .buddy-title {{
    font-family: 'Playfair Display', serif;
    color: #FFFFFF !important;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}}

.stApp .buddy-head .buddy-sub {{
    color: {CHAMPAGNE} !important;
    font-size: 0.85rem;
    letter-spacing: 0.8px;
    margin: 2px 0 0 0;
}}

/* Reward cards */
.reward-card {{
    background: linear-gradient(180deg, #FFFFFF, #FBF8F1);
    border: 1px solid rgba(201, 162, 75, 0.35);
    border-top: 4px solid {GOLD};
    border-radius: 18px;
    padding: 18px 18px;
    text-align: center;
    box-shadow: 0 10px 24px rgba(23, 75, 50, 0.08);
}}

.reward-card .big {{
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: {FOREST};
    line-height: 1.2;
}}

.reward-card .small {{
    font-size: 0.82rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #6B5B2A;
}}

.badge {{
    display: inline-block;
    background: {CHAMPAGNE};
    color: {FOREST};
    border: 1px solid {GOLD};
    border-radius: 999px;
    padding: 2px 12px;
    font-weight: 700;
    font-size: 0.85rem;
}}

/* Streamlit metrics as soft cards */
[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid rgba(201, 162, 75, 0.35);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: 0 6px 16px rgba(23, 75, 50, 0.06);
}}

[data-testid="stMetricValue"] {{
    font-family: 'Playfair Display', serif;
    color: {FOREST};
    font-weight: 800;
}}

/* Buttons */
div.stButton > button {{
    background: linear-gradient(135deg, {FOREST}, {EMERALD});
    color: #FFFFFF !important;
    border-radius: 12px;
    border: 1px solid {GOLD};
    min-height: 46px;
    font-weight: 600;
    letter-spacing: 0.4px;
    box-shadow: 0 6px 14px rgba(11, 46, 34, 0.22);
    transition: all 0.2s ease;
}}

div.stButton > button p {{
    color: #FFFFFF !important;
}}

div.stButton > button:hover {{
    background: linear-gradient(135deg, {GOLD}, #A8832F);
    border-color: {GOLD};
    color: #FFFFFF !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 20px rgba(201, 162, 75, 0.35);
}}

div.stButton > button:disabled {{
    background: #B9C7BE;
    border-color: #B9C7BE;
    box-shadow: none;
    color: #FFFFFF !important;
}}

/* Danger buttons (type="primary"): used for destructive confirmations */
div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #B93B3B, #8E2323);
    border: 1px solid #8E2323;
    box-shadow: 0 6px 14px rgba(142, 35, 35, 0.28);
}}

div.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #D64545, #A52A2A);
    border-color: #A52A2A;
    box-shadow: 0 10px 20px rgba(165, 42, 42, 0.35);
}}

/* Soft outlined button (type="secondary" inside the reset panel) */
.reset-panel {{
    background: linear-gradient(135deg, #FFF6F4, #FFFFFF);
    border: 1px solid rgba(185, 59, 59, 0.35);
    border-left: 6px solid #B93B3B;
    border-radius: 16px;
    padding: 16px 20px;
    margin: 10px 0 12px 0;
}}

.reset-panel .reset-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #8E2323;
    margin: 0 0 4px 0;
}}

.reset-panel .reset-text {{
    font-size: 0.92rem;
    color: {INK};
    margin: 0;
}}

/* ---------- INPUT TABS (upload / live camera) ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid rgba(201, 162, 75, 0.45);
}}

.stTabs [data-baseweb="tab"] {{
    font-weight: 600;
    color: {FOREST};
    border-radius: 10px 10px 0 0;
    padding: 8px 16px;
}}

.stTabs [aria-selected="true"] {{
    background: {PALE_LEAF};
    color: {FOREST};
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {GOLD};
}}

/* Camera preview */
[data-testid="stCameraInput"] {{
    border-radius: 16px;
    overflow: hidden;
}}

[data-testid="stCameraInput"] button {{
    border-radius: 12px;
}}

/* Batch counter pill */
.batch-pill {{
    display: inline-block;
    background: {CHAMPAGNE};
    color: {FOREST};
    border: 1px solid {GOLD};
    border-radius: 999px;
    padding: 3px 14px;
    font-weight: 700;
    font-size: 0.85rem;
    margin: 4px 0 8px 0;
}}

/* Expanders (one per analysed image) */
[data-testid="stExpander"] {{
    border: 1px solid rgba(201, 162, 75, 0.35) !important;
    border-radius: 16px !important;
    background: rgba(255, 255, 255, 0.7);
    margin-bottom: 12px;
}}

/* ---------- CUSTOM PROGRESS BAR (daily goal) ----------
   Only the used part is filled with dark green; the unused part keeps a
   soft track that blends with the page background. */
.pbar-wrap {{
    margin: 6px 0 14px 0;
}}

.pbar-label {{
    font-size: 0.95rem;
    color: {INK};
    margin-bottom: 6px;
}}

.pbar-track {{
    width: 100%;
    height: 12px;
    background: rgba(23, 75, 50, 0.07);
    border: 1px solid rgba(23, 75, 50, 0.10);
    border-radius: 999px;
    overflow: hidden;
}}

.pbar-fill {{
    height: 100%;
    background: linear-gradient(90deg, {FOREST}, #1F6F49);
    border-radius: 999px;
    transition: width 0.4s ease;
}}

/* ---------- VIRTUAL BIN CARDS ----------
   Each card gets its own accent colour through CSS variables:
   --accent (main), --tint (soft wash), --line (outline), --soft (fade). */
.bin-card {{
    padding: 4px 4px 2px 4px;
}}

.bin-head {{
    display: flex;
    align-items: center;
    gap: 14px;
    background: linear-gradient(135deg, var(--tint), rgba(255, 255, 255, 0));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 14px 16px;
}}

.bin-medal {{
    width: 54px;
    height: 54px;
    flex-shrink: 0;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    background: #FFFFFF;
    border: 2px solid var(--accent);
    box-shadow: 0 0 0 4px var(--tint),
                0 6px 14px rgba(11, 46, 34, 0.12);
}}

.bin-name {{
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: {FOREST};
    line-height: 1.1;
}}

.bin-id {{
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6B5B2A;
    margin-top: 5px;
}}

.bin-count-row {{
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin: 20px 4px 16px 4px;
}}

.bin-count {{
    font-family: 'Playfair Display', serif;
    font-size: 2.9rem;
    font-weight: 800;
    color: {INK};
    line-height: 1;
}}

.bin-count-label {{
    font-size: 0.74rem;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #5B6B60;
}}

.bin-cap-row {{
    display: flex;
    justify-content: space-between;
    margin: 0 4px 7px 4px;
    font-size: 0.74rem;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #5B6B60;
}}

.bin-cap-val {{
    font-weight: 700;
    color: {INK};
}}

.bin-track {{
    height: 10px;
    margin: 0 4px;
    background: rgba(23, 75, 50, 0.07);
    border: 1px solid rgba(23, 75, 50, 0.10);
    border-radius: 999px;
    overflow: hidden;
}}

.bin-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--soft));
    transition: width 0.4s ease;
}}

.bin-status {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin: 18px 4px 10px 4px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}}

.bin-status .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}}

.bin-status.ok {{
    background: #E3F3E8;
    color: #1F6F49;
}}

.bin-status.warn {{
    background: #FBF0CF;
    color: #8A6A12;
}}

.bin-status.full {{
    background: #FBE1E1;
    color: #9B2C2C;
}}

/* Upload area */
[data-testid="stFileUploader"] {{
    background-color: #FFFFFF;
    border-radius: 14px;
}}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small {{
    color: {INK} !important;
}}

/* Images */
[data-testid="stImage"] img {{
    border-radius: 16px;
    box-shadow: 0 8px 22px rgba(11, 46, 34, 0.18);
}}

/* Footer */
.footer {{
    text-align: center;
    color: {FOREST} !important;
    padding: 26px;
    margin-top: 26px;
    border-top: 1px solid rgba(201, 162, 75, 0.5);
    font-family: 'Playfair Display', serif;
    letter-spacing: 1px;
}}
</style>
""", unsafe_allow_html=True)


# ---------- PROGRESS BAR HELPER ----------
def progress_bar(percent, label=None):
    """Return HTML for a progress bar: dark green fill, soft empty track."""
    percent = max(0.0, min(float(percent), 100.0))
    label_html = f'<div class="pbar-label">{label}</div>' if label else ""
    return (
        '<div class="pbar-wrap">'
        f'{label_html}'
        '<div class="pbar-track">'
        f'<div class="pbar-fill" style="width:{percent}%"></div>'
        '</div>'
        '</div>'
    )


# ---------- BIN CARD HELPER ----------
def bin_card_html(name, data):
    """Return the HTML for one elegant bin card (no buttons)."""
    accent = BIN_COLORS.get(name, FOREST)
    icon = BIN_ICONS.get(name, "🗑️")
    capacity = max(0.0, min(float(data["capacity_percent"]), 100.0))
    count = data["item_count"]

    if capacity >= 90:
        status_class, status_text = "full", "Nearly full"
    elif capacity >= 70:
        status_class, status_text = "warn", "Filling up"
    else:
        status_class, status_text = "ok", "Available"

    style = (
        f"--accent:{accent};--tint:{accent}1F;"
        f"--line:{accent}55;--soft:{accent}B3;"
    )

    # Written without indentation or blank lines on purpose (markdown rule).
    return (
        f'<div class="bin-card" style="{style}">'
        '<div class="bin-head">'
        f'<div class="bin-medal">{icon}</div>'
        '<div>'
        f'<div class="bin-name">{name} Bin</div>'
        f'<div class="bin-id">{data["bin_id"]}</div>'
        '</div>'
        '</div>'
        '<div class="bin-count-row">'
        f'<div class="bin-count">{count}</div>'
        '<div class="bin-count-label">items recorded</div>'
        '</div>'
        '<div class="bin-cap-row">'
        '<span>Capacity</span>'
        f'<span class="bin-cap-val">{capacity:.0f}%</span>'
        '</div>'
        '<div class="bin-track">'
        f'<div class="bin-fill" style="width:{capacity}%"></div>'
        '</div>'
        f'<div class="bin-status {status_class}">'
        f'<span class="dot"></span>{status_text}'
        '</div>'
        '</div>'
    )


# ---------- DATABASE HELPERS ----------
def _existing_db_paths():
    """Every SQLite file this app might be using (rewards + recycling events)."""
    candidates = []

    for module in (rewards_module, data_base):
        for attr in ("DB_NAME", "DB_PATH", "DATABASE"):
            value = getattr(module, attr, None)
            if value and str(value) not in candidates:
                candidates.append(str(value))

    if "recycling.db" not in candidates:
        candidates.append("recycling.db")

    return [path for path in candidates if os.path.exists(path)]


def _table_columns(cur, table):
    return [col[1] for col in cur.execute(f'PRAGMA table_info("{table}")')]


# ---------- CLEAR RECYCLING EVENTS ----------
def clear_all_events():
    """Delete every saved recycling event from the SQLite database.

    Eco Points and streak tables are intentionally left untouched, because
    they do not contain both 'item_name' and 'material' columns.
    """
    db_path = (
        getattr(data_base, "DB_NAME", None)
        or getattr(data_base, "DB_PATH", None)
        or getattr(data_base, "DATABASE", None)
        or "recycling.db"
    )

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            columns = _table_columns(cur, table)
            # Only touch the table that stores recycling events
            if "item_name" in columns and "material" in columns:
                cur.execute(f'DELETE FROM "{table}"')
        conn.commit()
    finally:
        conn.close()


# ---------- CLEAR ECO POINTS / REWARDS ----------
def clear_all_rewards():
    """Permanently delete all Eco Points, streaks and rewarded events.

    The recycling-events table (the one with 'item_name' and 'material'
    columns) is left alone, so the virtual bins and the
    "Recent Recycling Events" list are NOT affected.

    1. If rewards.py has its own reset function, that is used.
    2. Otherwise every table that is not the recycling-events table is emptied.
    """
    # 1) Prefer a reset function provided by the rewards module itself
    for func_name in (
        "reset_rewards",
        "clear_rewards",
        "clear_all_rewards",
        "reset_all",
        "reset_points",
    ):
        reset_func = getattr(rewards_module, func_name, None)
        if callable(reset_func):
            reset_func()
            return

    # 2) Fallback: empty the rewards tables directly
    db_paths = _existing_db_paths()
    if not db_paths:
        raise FileNotFoundError("No database file was found.")

    cleared_tables = 0

    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            tables = [
                row[0]
                for row in cur.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            for table in tables:
                columns = _table_columns(cur, table)
                # Skip the recycling-events table
                if "item_name" in columns and "material" in columns:
                    continue
                cur.execute(f'DELETE FROM "{table}"')
                cleared_tables += 1
            conn.commit()
        finally:
            conn.close()

    if cleared_tables == 0:
        raise RuntimeError("No rewards tables were found to clear.")

    # Make sure the rewards tables exist and are ready for new points
    init_rewards_db()


# ---------- CHART HELPERS ----------
def bin_color_scale():
    return alt.Scale(
        domain=list(BIN_COLORS.keys()),
        range=list(BIN_COLORS.values())
    )


def colorful_bar_chart(data, y_title, height=300):
    """Bar chart with one colour per bin. `data` is {bin_name: value}."""
    df = pd.DataFrame(
        {"Category": list(data.keys()), "Value": list(data.values())}
    )
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Category:N", sort=None, title=None,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color("Category:N", scale=bin_color_scale(),
                            legend=None),
            tooltip=[
                alt.Tooltip("Category:N"),
                alt.Tooltip("Value:Q", title=y_title)
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def donut_chart(rows, value_key, height=300):
    """Donut chart. `rows` is a list of dicts with 'category' and value_key."""
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=70, outerRadius=120)
        .encode(
            theta=alt.Theta(f"{value_key}:Q"),
            color=alt.Color("category:N", scale=bin_color_scale(),
                            legend=alt.Legend(title="Category")),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip(f"{value_key}:Q", title=value_key.title()),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def daily_chart(history, value_key, color, y_title, height=300):
    """Last-7-days bar chart in a single colour."""
    df = pd.DataFrame(history)
    chart = (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("date:N", sort=None, title=None,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{value_key}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("date:N", title="Day"),
                alt.Tooltip(f"{value_key}:Q", title=y_title),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------- AI MODEL ----------

@st.cache_resource
def load_classifier():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-large-patch14"
    )


# ---------- IMAGE HELPERS ----------
def open_image(data):
    """Turn raw bytes into an RGB PIL image (raises if the file is not an image)."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def collect_images():
    """Gather every image the user has supplied, from files and from the camera.

    Returns (images, problems). Each image is a dict: {name, image}.
    """
    images = []
    problems = []

    # 1) Files picked in the uploader (can be many)
    for uploaded in st.session_state.get(
        f"uploader_{st.session_state.uploader_key}"
    ) or []:
        try:
            images.append({
                "name": uploaded.name,
                "image": open_image(uploaded.getvalue()),
            })
        except Exception:
            problems.append(uploaded.name)

    # 2) Photos taken live with the camera
    for shot in st.session_state.camera_shots:
        try:
            images.append({
                "name": shot["name"],
                "image": open_image(shot["data"]),
            })
        except Exception:
            problems.append(shot["name"])

    return images, problems


# ---------- RECYCLING GUIDANCE ----------
def get_recycling_advice(label):
    # Labels from the model use spaces ("cardboard box"); match on underscores
    label = normalize_label(label)

    # Sharps and medical waste need their own, stricter advice
    if any(word in label for word in SHARPS_KEYWORDS):
        return {
            "item": "Possible sharps or medical waste",
            "material": (
                "Needles, syringes and other medical waste are biohazards. "
                "They can injure people and spread infection, so they must "
                "never go in ordinary recycling or general waste."
            ),
            "prepare": [
                "Do not touch sharps with bare hands, and never recap, "
                "bend or break needles.",
                "Place them in a puncture-resistant sharps container "
                "(or a thick, rigid, sealable container if none is available).",
                "Keep the container upright, closed and out of reach of "
                "children and pets."
            ],
            "dispose": [
                "Hand the sealed container to a hospital, clinic, pharmacy "
                "or authorised biomedical-waste service.",
                "Never put loose sharps in the regular bin or recycling.",
                "If someone is pricked or cut, wash the area with soap and "
                "water and seek medical advice promptly."
            ],
            "reuse": [
                "Sharps and used medical items must not be reused.",
                "Ask your clinic or pharmacy about a sharps take-back "
                "programme."
            ]
        }

    # Other hazardous items
    if any(word in label for word in HAZARDOUS_KEYWORDS):
        return {
            "item": "Possible hazardous item",
            "material": (
                "Batteries, bulbs, medicines, chemicals, aerosols and paints "
                "need special handling and should not go in ordinary "
                "recycling or general waste."
            ),
            "prepare": [
                "Keep the item intact; do not crush, puncture or open it.",
                "Store it in a dry place away from heat and children.",
                "Keep it in its original container and do not mix "
                "different chemicals together."
            ],
            "dispose": [
                "Use a designated hazardous-waste or take-back collection point.",
                "Never place it in the regular bin or pour it down the drain.",
                "Ask your local waste service if you are unsure."
            ],
            "reuse": [
                "Rechargeable batteries can be reused many times.",
                "Unused medicines and chemicals should go to a take-back "
                "programme, not be passed on."
            ]
        }

    elif "food" in label or "peel" in label:
        return {
            "item": "Possible biodegradable item",
            "material": (
                "Food and plant waste can usually be composted, "
                "but cooked food or meat may have separate rules."
            ),
            "prepare": [
                "Remove any packaging, stickers or plastic.",
                "Drain excess liquid.",
                "Keep it separate from non-biodegradable waste."
            ],
            "dispose": [
                "Place it in a compost or organic-waste bin.",
                "Check local rules for cooked food and meat or dairy.",
                "Do not put compostable waste in recycling bins."
            ],
            "reuse": [
                "Compost it at home for garden use.",
                "Some peels and scraps can be reused in cooking "
                "or as plant fertiliser."
            ]
        }

    elif "plastic_bottle" in label or "plastic_cover" in label:
        return {
            "item": "Possible bottle or cover",
            "material": (
                "Not confirmed. Check the recycling symbol "
                "or material label on the item."
            ),
            "prepare": [
                "Empty any remaining liquid.",
                "Rinse if needed and let the container drain.",
                "Check whether the cap and label need separate sorting."
            ],
            "dispose": [
                "Check whether your local service accepts this type "
                "of container.",
                "Follow the instructions on your local recycling bins.",
                "If the material is unknown, ask your local waste service."
            ],
            "reuse": [
                "If the container is designed for reuse, clean it "
                "and reuse it as intended.",
                "Avoid reusing disposable containers for purposes "
                "they were not designed for."
            ]
        }

    elif "metal_can" in label or "tin" in label:
        return {
            "item": "Possible metal can",
            "material": (
                "Could be metal, but verify the item before sorting."
            ),
            "prepare": [
                "Empty the contents.",
                "Rinse away leftover food or drink when practical.",
                "Handle sharp edges carefully."
            ],
            "dispose": [
                "Check whether your local service accepts metal cans.",
                "Follow local instructions for placing cans in bins.",
                "Do not place hazardous chemical containers in "
                "ordinary recycling."
            ],
            "reuse": [
                "Consider reuse only if the container is safe "
                "and has no sharp edges.",
                "Keep food-contact reuse limited to containers "
                "designed for that purpose."
            ]
        }

    elif "paper" in label or "cardboard_box" in label:
        return {
            "item": "Possible paper-based item",
            "material": (
                "Paper, coated paper and composite cartons may "
                "need different recycling routes."
            ),
            "prepare": [
                "Keep paper clean and dry.",
                "Remove non-paper parts when practical and required.",
                "Flatten cardboard boxes to save space."
            ],
            "dispose": [
                "Check whether coated or laminated paper is accepted.",
                "Check whether cartons have a separate collection route.",
                "Keep heavily soiled paper out of paper recycling "
                "unless local guidance says otherwise."
            ],
            "reuse": [
                "Reuse clean paper for notes or crafts.",
                "Reuse sturdy cardboard boxes for storage."
            ]
        }

    elif "plastic" in label:
        return {
            "item": "Possible plastic item",
            "material": (
                "The plastic resin type is unknown. Look for "
                "the recycling code or material marking."
            ),
            "prepare": [
                "Empty the item first.",
                "Rinse it if required by your local collection rules.",
                "Check whether lids or other parts need separate sorting."
            ],
            "dispose": [
                "Do not assume every plastic type is accepted.",
                "Check the local rules for the specific plastic code.",
                "Use a designated collection point if required."
            ],
            "reuse": [
                "Reuse durable items when safe and practical.",
                "Avoid reusing packaging for food or drink unless "
                "it is intended for that purpose."
            ]
        }

    elif "glass" in label or "jar" in label:
        return {
            "item": "Possible glass item",
            "material": (
                "Verify whether it is a recyclable glass container "
                "or another type of glass."
            ),
            "prepare": [
                "Empty the container.",
                "Rinse it if needed.",
                "Handle broken glass carefully; never place loose "
                "shards in a bin without following local instructions."
            ],
            "dispose": [
                "Check whether your local service accepts glass.",
                "Follow local rules for colour-separated glass, if any.",
                "Do not assume mirrors, ceramics or heat-resistant "
                "glass belong in bottle-and-jar recycling."
            ],
            "reuse": [
                "Reuse intact jars or containers when suitable.",
                "Do not reuse chipped or cracked glass containers."
            ]
        }

    else:
        return {
            "item": "Item not confidently categorized",
            "material": (
                "The image prediction does not establish the "
                "item's material."
            ),
            "prepare": [
                "Inspect the item and its material label.",
                "Look for a recycling symbol or product markings.",
                "Avoid guessing when the material is unclear."
            ],
            "dispose": [
                "Use a manual material selection if available.",
                "Check your local waste sorting guidelines.",
                "Use a designated collection point if the item "
                "requires special handling."
            ],
            "reuse": [
                "Consider reuse if the item is safe and functional.",
                "Check whether a suitable repair or take-back "
                "option exists."
            ]
        }


# ---------- RECYCLING BIN HELPERS ----------
def record_and_reward(item, category):
    """Record one analysed image in its recycling bin and award its Eco Points.

    Returns a (status, detail) tuple. status is one of:
      "ok"         recorded and rewarded
      "rewarded"   recorded, but the database said it was already rewarded
      "points_err" recorded, but saving points failed (detail = error text)
      "duplicate"  this image was already recorded earlier
      "full"       the bin could not accept the item
    """
    uid = item["uid"]

    if uid in st.session_state.recorded_uids:
        return "duplicate", ""

    recorded = record_item(
        category,
        item["best_label"],
        st.session_state.digital_twin_bins
    )

    if not recorded:
        return "full", ""

    st.session_state.recorded_uids.add(uid)
    # Remember which bin this item went into, so the card can show it later
    st.session_state.recorded_bins[uid] = category

    # Points are awarded ONLY after the event was recorded.
    try:
        awarded = award_event(
            event_key=uid,
            item_label=item["best_label"],
            bin_name=category
        )
    except Exception as error:
        return "points_err", str(error)

    return ("ok" if awarded else "rewarded"), ""


def suggested_category(item, bin_names):
    """Best-guess bin for an analysed image."""
    suggested = resolve_bin(item["best_label"])
    return suggested if suggested in bin_names else bin_names[0]


# ---------- ONE ANALYSED IMAGE (result card) ----------
def render_result(item, number, total, bin_status):
    """Show the full AI analysis, ReWA Buddy and action plan for one image."""
    best_label = item["best_label"]
    best_score = item["best_score"]
    uid = item["uid"]

    title = f"{number}. {item['name']} → {best_label.title()} ({best_score:.0%})"

    with st.expander(title, expanded=(total == 1 or number == 1)):
        # Wide layout: photo on the left, verdict on the right
        photo_col, verdict_col = st.columns([1, 2], gap="large")

        with photo_col:
            st.image(
                item["image"],
                caption=item["name"],
                use_container_width=True
            )

        with verdict_col:
            if best_score < 0.55:
                st.warning(
                    "The model score is low. Try a clearer photo "
                    "or verify the item manually."
                )
            else:
                st.success(
                    f"Best guess: {best_label.title()} "
                    f"({best_score:.1%} model score)"
                )

            if is_hazardous_label(best_label):
                st.error(
                    "☣️ This looks like hazardous waste. Handle it with care "
                    "and do not put it in the regular bin."
                )

            st.markdown(
                f'<div class="result">'
                f'<h3>🏷️ {best_label.title()}</h3>'
                f'<p>Model score: {best_score:.1%}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write("**Other possible predictions**")
            for other in item["results"][1:]:
                st.write(
                    f"• {other['label'].title()} — "
                    f"{other['score']:.1%}"
                )

        # ---- ReWA AI Buddy ----
        st.markdown(
            '<div class="buddy-head">'
            '<div class="buddy-title">🌿 ReWA AI Buddy</div>'
            '<p class="buddy-sub">RECYCLING &amp; REUSE ADVISOR</p>'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "✨ Ask ReWA for Recycling & Reuse Advice",
            use_container_width=True,
            key=f"buddy_btn_{uid}"
        ):
            try:
                with st.spinner("ReWA is preparing your recycling advice..."):
                    st.session_state.buddy_advice[uid] = generate_recycling_advice(
                        waste_type=best_label,
                        confidence=best_score,
                        bin_status=bin_status
                    )
            except Exception as error:
                st.session_state.buddy_advice.pop(uid, None)
                st.warning(
                    "ReWA AI Buddy is temporarily unavailable. "
                    "Your general recycling guidance is still available."
                )
                st.caption(f"Technical details: {error}")

        if st.session_state.buddy_advice.get(uid):
            with st.container(border=True):
                st.markdown(st.session_state.buddy_advice[uid])

        # ---- Action plan ----
        advice = get_recycling_advice(best_label)

        st.markdown("## ♻️ Your Recycling Action Plan")
        st.write(f"**Item assessment:** {advice['item']}")

        st.markdown("### 🧪 Material check")
        st.write(advice["material"])

        st.markdown("### 🧼 Step 1: Prepare the item")
        for index, step in enumerate(advice["prepare"]):
            st.checkbox(
                step,
                key=f"prepare_{uid}_{index}"
            )

        st.markdown("### 🗑️ Step 2: Dispose responsibly")
        for step in advice["dispose"]:
            st.markdown(f"- {step}")

        st.markdown("### 🌱 Step 3: Reuse ideas")
        for step in advice["reuse"]:
            st.markdown(f"- {step}")

        st.info(
            "These are general suggestions based on the AI's predicted "
            "label. Verify the actual material and follow local recycling "
            "rules before disposal."
        )


# ---------- HEADER (dark emerald hero banner) ----------
banner_mime, banner_b64 = (None, None)
if USE_BANNER_IMAGE:
    banner_mime, banner_b64 = load_banner_base64()

# NOTE: the HTML below is deliberately written without indentation or blank
# lines; Streamlit's markdown would otherwise treat it as a code block.
if banner_b64:
    header_html = (
        '<div class="logo-banner image-mode">'
        f'<img class="logo-img" src="data:{banner_mime};base64,{banner_b64}" '
        'alt="ReWA.AI - Recycling-Reusable Waste Assistant">'
        '</div>'
    )
else:
    header_html = (
        '<div class="logo-banner">'
        '<div class="globe-wrap"><div class="globe"></div>'
        f'{GLOBE_SVG}'
        '</div>'
        f'{TILES_HTML}'
        '<div class="banner-center">'
        f'{LOGO_MARK_SVG}'
        '<div class="wordmark">'
        '<span class="w-white">ReW</span>'
        f'<span class="w-green"><span class="a-wrap">A{WORD_LEAF_SVG}</span>.AI</span>'
        '</div>'
        '<div class="full-name">(Recycling-Reusable Waste Assistant)</div>'
        '<div class="rule">'
        '<span class="line"></span>'
        f'{DIVIDER_LEAF_SVG}'
        '<span class="line r"></span>'
        '</div>'
        '<div class="tagline">Sort Today<span class="dot">•</span>Save Tomorrow</div>'
        '</div>'
        '</div>'
    )

st.markdown(header_html, unsafe_allow_html=True)

# One-time message left by the previous run (e.g. after recording an item)
if st.session_state.flash:
    flash_kind, flash_text = st.session_state.flash
    st.session_state.flash = None
    if flash_kind == "success":
        st.success(flash_text)
    else:
        st.warning(flash_text)

# ---------- ECO REWARDS ----------
rewards = get_summary()

st.header("🏆 Your Eco Rewards")

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.markdown(
        f'<div class="reward-card"><div class="big">🌟 {rewards["total_points"]}</div>'
        f'<div class="small">Eco Points</div></div>',
        unsafe_allow_html=True
    )

with r2:
    day_word = "day" if rewards["current_streak"] == 1 else "days"
    st.markdown(
        f'<div class="reward-card"><div class="big">🔥 {rewards["current_streak"]} {day_word}</div>'
        f'<div class="small">Current streak</div></div>',
        unsafe_allow_html=True
    )

with r3:
    st.markdown(
        f'<div class="reward-card"><div class="big">🏅 {rewards["longest_streak"]}</div>'
        f'<div class="small">Longest streak (days)</div></div>',
        unsafe_allow_html=True
    )

with r4:
    st.markdown(
        f'<div class="reward-card"><div class="big">📅 {rewards["active_days"]}</div>'
        f'<div class="small">Active recycling days</div></div>',
        unsafe_allow_html=True
    )

st.write("")

goal_percent = min(int(rewards["today_events"] / DAILY_GOAL * 100), 100)
st.markdown(
    progress_bar(
        goal_percent,
        f"Today's goal: {rewards['today_events']} of {DAILY_GOAL} items "
        f"recycled • {rewards['today_points']} points earned today"
    ),
    unsafe_allow_html=True
)

if rewards["today_events"] == 0 and rewards["current_streak"] > 0:
    st.caption(
        f"Recycle an item today to keep your {rewards['current_streak']}-day "
        f"streak alive!"
    )
elif rewards["today_events"] >= DAILY_GOAL:
    st.caption("🎉 Daily goal complete — great work!")

st.caption(f"Each recorded recycling event earns {POINTS_PER_EVENT} Eco Points.")

# ---- Reset Eco Points (two-step confirmation) ----
has_rewards = (
    rewards["total_points"] > 0
    or rewards["total_events"] > 0
    or rewards["active_days"] > 0
)

if not st.session_state.confirm_reset_rewards:
    # Step 1: the normal button
    reset_btn_col, _reset_space = st.columns([1, 3])
    with reset_btn_col:
        if st.button(
            "🗑️ Reset Eco Points",
            use_container_width=True,
            disabled=not has_rewards,
            key="reset_rewards_start",
            help="Delete all Eco Points and streaks (asks for confirmation first)."
        ):
            st.session_state.confirm_reset_rewards = True
            st.rerun()
else:
    # Step 2: ask once more before anything is deleted
    st.markdown(
        '<div class="reset-panel">'
        '<div class="reset-title">⚠️ Are you absolutely sure?</div>'
        '<p class="reset-text">This will permanently delete '
        f'<b>{rewards["total_points"]} Eco Points</b>, your streaks and your '
        'rewarded-event history. This cannot be undone.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    confirm_col, cancel_col, _confirm_space = st.columns([1.3, 1, 2])

    with confirm_col:
        if st.button(
            "✅ Yes, delete my Eco Points",
            type="primary",
            use_container_width=True,
            key="reset_rewards_confirm"
        ):
            try:
                clear_all_rewards()
                st.session_state.confirm_reset_rewards = False
                st.session_state.flash = (
                    "success",
                    "🧹 All Eco Points and streaks have been reset."
                )
            except Exception as error:
                st.session_state.confirm_reset_rewards = False
                st.session_state.flash = (
                    "warning",
                    f"Could not reset Eco Points: {error}"
                )
            st.rerun()

    with cancel_col:
        if st.button(
            "↩️ Cancel",
            use_container_width=True,
            key="reset_rewards_cancel"
        ):
            st.session_state.confirm_reset_rewards = False
            st.rerun()

st.markdown("---")

st.title("Welcome to a greener future!")
st.write(
    "Upload photos or take live pictures of your items and let AI suggest "
    "what they might be. You'll also receive practical recycling guidance "
    "for every item."
)

# Bins are needed by the ReWA AI Buddy inside the analysis cards
bins = st.session_state.digital_twin_bins

# ---------- MAIN LAYOUT (stacked: full-width upload, then full-width analysis) ----------
upload_row = st.container()
result_row = st.container()

with upload_row:
    with st.container(border=True):
        st.subheader("📷 1. Add your images")
        st.write(
            "Upload one or many photos, or snap them live with your camera."
        )

        tab_upload, tab_camera = st.tabs(
            ["📁 Upload files", "📸 Take live photo"]
        )

        # ---- Tab 1: upload one or many files ----
        with tab_upload:
            st.file_uploader(
                "Choose photos of your items (you can select several)",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.uploader_key}"
            )

        # ---- Tab 2: live camera ----
        with tab_camera:
            st.caption(
                "Allow camera access in your browser when asked. "
                "Take a picture, then press “Add this photo”. "
                "Repeat for as many items as you like."
            )

            camera_photo = st.camera_input(
                "Take a picture of the item",
                key=f"camera_{st.session_state.camera_key}"
            )

            if camera_photo is not None:
                if st.button(
                    "➕ Add this photo to the batch",
                    use_container_width=True,
                    key="add_camera_photo"
                ):
                    photo_bytes = camera_photo.getvalue()
                    photo_id = hashlib.md5(photo_bytes).hexdigest()

                    if any(s["id"] == photo_id
                           for s in st.session_state.camera_shots):
                        st.info("This photo is already in the batch.")
                    else:
                        st.session_state.camera_shots.append({
                            "id": photo_id,
                            "name": f"Camera photo "
                                    f"{len(st.session_state.camera_shots) + 1}",
                            "data": photo_bytes,
                        })
                        # Fresh camera widget, ready for the next picture
                        st.session_state.camera_key += 1
                        st.session_state.flash = (
                            "success",
                            "📸 Photo added! Take another one or press "
                            "“Classify” to analyse your batch."
                        )
                        st.rerun()

            if st.session_state.camera_shots:
                st.markdown(
                    f'<span class="batch-pill">📸 '
                    f'{len(st.session_state.camera_shots)} live photo(s) '
                    f'in the batch</span>',
                    unsafe_allow_html=True
                )
                if st.button(
                    "🗑️ Remove all live photos",
                    use_container_width=True,
                    key="remove_camera_photos"
                ):
                    st.session_state.camera_shots = []
                    st.session_state.camera_key += 1
                    st.rerun()

        # ---- Everything collected so far ----
        images, problems = collect_images()

        for bad_name in problems:
            st.error(f"Unable to open “{bad_name}”. Try another file.")

        if len(images) > MAX_IMAGES:
            st.warning(
                f"You added {len(images)} images. Only the first "
                f"{MAX_IMAGES} will be analysed at a time."
            )
            images = images[:MAX_IMAGES]

        if images:
            st.markdown(
                f'<span class="batch-pill">🖼️ {len(images)} image(s) '
                f'ready to classify</span>',
                unsafe_allow_html=True
            )
            st.image(
                [img["image"] for img in images],
                caption=[img["name"] for img in images],
                width=130
            )

        button_left, button_right = st.columns([3, 1])

        with button_left:
            classify_clicked = st.button(
                f"🔍 Classify {len(images)} Image(s)" if images
                else "🔍 Classify Images",
                use_container_width=True,
                disabled=len(images) == 0
            )

        with button_right:
            if st.button(
                "🧹 Clear all",
                use_container_width=True,
                disabled=(len(images) == 0 and not st.session_state.batch)
            ):
                st.session_state.camera_shots = []
                st.session_state.batch = []
                st.session_state.buddy_advice = {}
                st.session_state.uploader_key += 1
                st.session_state.camera_key += 1
                st.rerun()

        st.caption("Supported formats: JPG, PNG and WEBP")

with result_row:
    with st.container(border=True):
        st.subheader("✨ 2. AI analysis")

        # Run classification only when the button is clicked
        if classify_clicked and images:
            try:
                classifier = load_classifier()
                new_batch = []
                progress = st.progress(0.0, text="Analyzing your images...")

                for position, item in enumerate(images, start=1):
                    try:
                        predictions = classifier(
                            item["image"],
                            candidate_labels=CANDIDATE_LABELS
                        )[:3]

                        new_batch.append({
                            # Unique key for this classification; the database
                            # uses it to make sure the same event can never
                            # be rewarded twice.
                            "uid": uuid.uuid4().hex,
                            "name": item["name"],
                            "image": item["image"],
                            "results": predictions,
                            "best_label": predictions[0]["label"],
                            "best_score": predictions[0]["score"],
                        })
                    except Exception as error:
                        st.error(
                            f"Could not analyze {item['name']}: {error}"
                        )

                    progress.progress(
                        position / len(images),
                        text=f"Analyzed {position} of {len(images)} image(s)"
                    )

                progress.empty()

                st.session_state.batch = new_batch
                # New items mean the old ReWA answers no longer apply
                st.session_state.buddy_advice = {}

            except Exception as error:
                st.error(f"Could not load the AI model: {error}")

        # ---- Saved results, shown directly under "AI analysis" ----
        batch = st.session_state.batch

        if not batch:
            st.write(
                "Your AI predictions and recycling advice will appear here."
            )
        else:
            if len(batch) > 1:
                st.markdown(
                    f'<span class="batch-pill">✅ {len(batch)} images '
                    f'analysed</span>',
                    unsafe_allow_html=True
                )

            # Real values from the virtual bins
            bin_status = {
                name.lower(): {
                    "fill_percent": data["capacity_percent"],
                    "source": "simulated"
                }
                for name, data in bins.items()
            }

            for number, item in enumerate(batch, start=1):
                render_result(item, number, len(batch), bin_status)


# ---------- VIRTUAL RECYCLING STATION (sorting section) ----------
st.markdown("---")
st.header("♻️ Virtual Recycling Station")

batch = st.session_state.batch
bin_names = list(st.session_state.digital_twin_bins.keys())

if batch:
    pending = [i for i in batch if i["uid"] not in st.session_state.recorded_uids]

    st.write(
        f"**{len(batch)}** item(s) detected • "
        f"**{len(pending)}** waiting to be added."
    )

    # ---- Add everything in one click ----
    if len(batch) > 1:
        if st.button(
            f"➕ Add all {len(pending)} pending items "
            f"(+{POINTS_PER_EVENT} Eco Points each)",
            disabled=len(pending) == 0,
            use_container_width=True,
            key="add_all_items"
        ):
            ok_count = 0
            problems_found = []

            for item in pending:
                # Use the bin currently shown in each item's dropdown
                chosen = st.session_state.get(
                    f"confirmed_category_{item['uid']}",
                    suggested_category(item, bin_names)
                )
                status, detail = record_and_reward(item, chosen)

                if status in ("ok", "rewarded"):
                    ok_count += 1
                elif status == "points_err":
                    ok_count += 1
                    problems_found.append(
                        f"{item['best_label']}: points not saved ({detail})"
                    )
                elif status == "full":
                    problems_found.append(
                        f"{item['best_label']}: the {chosen} bin may be full"
                    )

            if problems_found:
                st.session_state.flash = (
                    "warning",
                    f"{ok_count} item(s) recorded. "
                    + " • ".join(problems_found)
                )
            else:
                st.session_state.flash = (
                    "success",
                    f"🌟 {ok_count} item(s) recorded! "
                    f"+{ok_count * POINTS_PER_EVENT} Eco Points."
                )
            st.rerun()

    # ---- One row per image ----
    for number, item in enumerate(batch, start=1):
        uid = item["uid"]
        best_label = item["best_label"]
        already = uid in st.session_state.recorded_uids

        with st.container(border=True):
            thumb_col, info_col = st.columns([1, 4], gap="medium")

            with thumb_col:
                st.image(item["image"], width=90)

            with info_col:
                st.write(f"**{number}. {item['name']}** — detected: **{best_label}**")

                # Pre-select the bin that matches the detected item
                # (hazardous items always go to the Hazardous bin)
                suggested = resolve_bin(best_label)
                if suggested is None:
                    st.caption(
                        "Couldn't auto-match this item. "
                        "Please choose the bin manually."
                    )
                default_index = (
                    bin_names.index(suggested) if suggested in bin_names else 0
                )

                category = st.selectbox(
                    "Confirm the waste category",
                    bin_names,
                    index=default_index,
                    key=f"confirmed_category_{uid}",
                    disabled=already
                )

                if already:
                    recorded_bin = st.session_state.recorded_bins.get(uid, category)
                    st.success(
                        f"✅ Already recorded in the {recorded_bin} recycling bin."
                    )
                elif st.button(
                    f"Add item to recycling bin (+{POINTS_PER_EVENT} Eco Points)",
                    key=f"add_item_{uid}"
                ):
                    status, detail = record_and_reward(item, category)

                    if status == "ok":
                        st.session_state.flash = (
                            "success",
                            f"🌟 +{POINTS_PER_EVENT} Eco Points! {best_label} "
                            f"recorded in the {category} bin."
                        )
                        st.rerun()
                    elif status == "rewarded":
                        st.session_state.flash = (
                            "warning",
                            "This recycling event was already rewarded."
                        )
                        st.rerun()
                    elif status == "points_err":
                        st.session_state.flash = (
                            "warning",
                            f"{best_label} was recorded, but points could "
                            f"not be saved: {detail}"
                        )
                        st.rerun()
                    elif status == "duplicate":
                        st.warning(
                            "This item has already been recorded in a "
                            "recycling bin."
                        )
                    else:
                        st.error(
                            f"Could not record the item. "
                            f"The {category} bin may be full."
                        )
else:
    st.info("Add photos (upload or live camera) and classify them to get started.")

# ---------- RECYCLING ANALYTICS ----------
st.markdown("---")
st.header("📊 Recycling Analytics")

bins = st.session_state.digital_twin_bins

# 1. Calculate the total recorded items
total_items = sum(
    data["item_count"]
    for data in bins.values()
)

# 2. Find the most frequently recorded category
if total_items > 0:
    most_common = max(
        bins,
        key=lambda name: bins[name]["item_count"]
    )
    common_count = bins[most_common]["item_count"]
else:
    most_common = "None yet"
    common_count = 0

# 3. Calculate average simulated capacity
average_capacity = sum(
    data["capacity_percent"]
    for data in bins.values()
) / len(bins)

# 4. Display summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total recorded items", total_items)

with col2:
    st.metric("Most recorded category", most_common)

with col3:
    st.metric("Average simulated capacity", f"{average_capacity:.1f}%")

# 5. Compare recorded item counts and simulated capacities side by side
chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Items by Material")
    colorful_bar_chart(
        {name: data["item_count"] for name, data in bins.items()},
        y_title="Items recorded"
    )

with chart_right:
    st.subheader("Virtual Bin Capacity Comparison")
    colorful_bar_chart(
        {name: data["capacity_percent"] for name, data in bins.items()},
        y_title="Simulated capacity (%)"
    )

# 6. Rewards analytics (read from the database, so it survives restarts)
st.subheader("🌟 Rewards Analytics")

# Re-read the summary so points recorded earlier in this run are included
rewards = get_summary()

s1, s2, s3, s4 = st.columns(4)
s1.metric("Total Eco Points", rewards["total_points"])
s2.metric("Rewarded events", rewards["total_events"])
s3.metric("Current streak", f"{rewards['current_streak']} days")
s4.metric("Longest streak", f"{rewards['longest_streak']} days")

points_by_category = get_points_by_category()
history = get_daily_history(days=7)

donut_col, daily_col = st.columns(2)

with donut_col:
    st.markdown("**Points earned by category**")
    if points_by_category:
        donut_chart(points_by_category, value_key="points")
    else:
        st.info("Record an item to see your points by category.")

with daily_col:
    st.markdown("**Daily progress — items recycled (last 7 days)**")
    if rewards["total_events"] > 0:
        daily_chart(
            history,
            value_key="events",
            color="#2E8B57",
            y_title="Items recycled"
        )
    else:
        st.info("Your daily progress will appear after your first item.")

if rewards["total_events"] > 0:
    st.markdown("**Eco Points earned per day (last 7 days)**")
    daily_chart(
        history,
        value_key="points",
        color="#C9A24B",
        y_title="Eco Points",
        height=240
    )

# ---------- RECENT RECYCLING EVENTS ----------
st.markdown("---")
st.subheader("🕘 Recent Recycling Events")

events = get_events()

if events:
    for event in events[:10]:
        st.write(
            f"♻️ {event['item_name']} — "
            f"{event['material']} — "
            f"{event['timestamp']}"
        )

    confirm_clear = st.checkbox(
        "Yes, permanently delete all recycling events",
        key="confirm_clear_events"
    )

    if st.button("🧹 Clear recent events", disabled=not confirm_clear):
        try:
            clear_all_events()
            st.toast("All recycling events cleared!")
            st.rerun()
        except Exception as error:
            st.error(f"Could not clear events: {error}")
else:
    st.info("No recycling events recorded yet.")

# ---------- DISPLAY VIRTUAL BINS ----------
st.markdown("---")
st.subheader("🏭 Your Recycling Bins")
st.caption(
    "Six dedicated bins, tracked live. "
    "Capacity is simulated, not sensor-measured."
)

bins = st.session_state.digital_twin_bins


def display_bin(name, data):
    with st.container(border=True):
        st.markdown(bin_card_html(name, data), unsafe_allow_html=True)

        if st.button(
            f"🗑️ Empty {name} bin",
            key=f"empty_{name}",
            use_container_width=True
        ):
            empty_bin(name, st.session_state.digital_twin_bins)
            st.rerun()


# Display all bins, three per row
bin_list = list(bins.keys())
for i in range(0, len(bin_list), 3):
    row_cols = st.columns(3, gap="medium")
    for row_col, name in zip(row_cols, bin_list[i:i + 3]):
        with row_col:
            display_bin(name, bins[name])

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    🌱 ReWA.AI &nbsp;•&nbsp; Reduce waste • Reuse • Recycle
    <br>
    Right item. Right bin. A cleaner tomorrow.
</div>
""", unsafe_allow_html=True)
