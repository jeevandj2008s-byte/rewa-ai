import base64
import hashlib
import io
import os
import random
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

# ---------- GREENERY PALETTE ----------
# Forest, emerald, sage and cream. (The CSS below defines the same colours as
# CSS variables; these Python constants are used by charts and bin cards.)
DEEP = "#06241A"        # deepest forest: intro, hero, footer
EMERALD = "#0C4A34"     # emerald: gradients
FOREST = "#14432F"      # primary: headings
LEAF = "#2E7D52"        # mid green
SAGE = "#9DB89B"        # soft sage
GOLD = "#C9A24B"        # accent: hairlines, highlights
CHAMPAGNE = "#F1E3BC"   # soft gold
MINT = "#A8E6C0"        # success / glow
IVORY = "#F7F3E7"       # cream page background
PALE_LEAF = "#E6EFE0"   # cards, subtle panels
INK = "#12241A"         # body text

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
# The hero is drawn with HTML/CSS/SVG, so it stays sharp on every screen.
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


# ---------- GRAPHICS (inline SVG) ----------
# Recycling arrows + leaf mark used in the intro and the hero.
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

# Same mark with its own gradient ids, so it can appear twice on one page.
INTRO_MARK_SVG = LOGO_MARK_SVG.replace("mkA", "iA").replace("mkB", "iB")

# Small leaf used in the divider line.
DIVIDER_LEAF_SVG = (
    '<svg class="rule-leaf" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M3 21 C3 9 11 3 22 3 C22 14 16 21 3 21 Z" fill="#E3F8D0"/>'
    '<path d="M3 21 L15 9" stroke="#1F6F49" stroke-width="1.4" fill="none" '
    'stroke-linecap="round"/>'
    '</svg>'
)

# A single leaf shape; its colour comes from CSS (fill: currentColor).
LEAF_SHAPE_SVG = (
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M3 21 C3 9 11 3 22 3 C22 14 16 21 3 21 Z"/>'
    '<path d="M3 21 L15 9" stroke="rgba(255,255,255,.45)" stroke-width="1.1" '
    'fill="none" stroke-linecap="round"/>'
    '</svg>'
)

LEAF_COLORS = ["#2E7D52", "#4E9A68", "#9DB89B", "#14432F", "#7CB98A", "#C9A24B"]
INTRO_LEAF_COLORS = ["#A8E6C0", "#D8F8B6", "#8FDB7A", "#4E9A68", "#F1E3BC"]
HERO_LEAF_COLORS = ["#A8E6C0", "#D8F8B6", "#8FDB7A", "#6CC08B", "#F1E3BC"]


def leaf_field(count, seed, size=(18, 44), duration=(18, 34),
               delay=(-30, 0), colors=None, spread_y=False):
    """Return `count` leaf <span>s with randomised (but repeatable) positions.

    Each leaf gets CSS variables (--x, --y, --s, --d, --dl, --c); the CSS
    decides how it moves (drifting, rising or floating).
    """
    rnd = random.Random(seed)
    colors = colors or LEAF_COLORS
    parts = []
    for _ in range(count):
        style = f"--x:{rnd.randint(2, 96)}%;"
        if spread_y:
            style += f"--y:{rnd.randint(4, 86)}%;"
        style += (
            f"--s:{rnd.randint(*size)}px;"
            f"--d:{rnd.uniform(*duration):.1f}s;"
            f"--dl:{rnd.uniform(*delay):.1f}s;"
            f"--c:{rnd.choice(colors)}"
        )
        parts.append(f'<span class="leaf" style="{style}">{LEAF_SHAPE_SVG}</span>')
    return "".join(parts)


def fireflies(count, seed):
    """Tiny glowing dots that twinkle (decorative)."""
    rnd = random.Random(seed)
    return "".join(
        f'<span class="fly" style="--x:{rnd.randint(3, 97)}%;'
        f'--y:{rnd.randint(5, 95)}%;--d:{rnd.uniform(2.5, 5):.1f}s;'
        f'--dl:{rnd.uniform(0, 3):.1f}s"></span>'
        for _ in range(count)
    )


# Four outlined icon tiles on the right-hand side of the hero.
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

# The cinematic intro plays once per browser session (first run only).
# FIX: the hero's "wait for the intro" delay must apply ONLY on that first
# run. Before, it stayed True for every rerun, so after every click the hero
# text stayed invisible for ~4 seconds and the page felt frozen.
show_intro = not st.session_state.get("intro_played", False)
if show_intro:
    st.session_state.intro_played = True
hero_after_intro = show_intro

# ---------- THEME (CSS ONLY: Streamlit strips <script> from markdown) ----------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700;9..144,800&family=Manrope:wght@500;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* =====================================================================
   DESIGN TOKENS
   ===================================================================== */
:root {
    --deep: #06241A;
    --emerald: #0C4A34;
    --forest: #14432F;
    --leaf: #2E7D52;
    --fern: #4E9A68;
    --sage: #9DB89B;
    --sage-soft: #DDE8D8;
    --mist: #EEF3E8;
    --cream: #F7F3E7;
    --paper: #FCFAF3;
    --gold: #C9A24B;
    --gold-soft: #F1E3BC;
    --ink: #12241A;
    --muted: #566B5E;
    --ease: cubic-bezier(.2, .8, .2, 1);
    --font-display: 'Fraunces', 'Playfair Display', Georgia, serif;
    --font-body: 'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

html,
[data-testid="stMain"],
section.main {
    scroll-behavior: smooth;
}

/* =====================================================================
   PAGE
   ===================================================================== */
.stApp {
    background:
        radial-gradient(circle at 6% 0%, rgba(157, 184, 155, 0.35), transparent 42%),
        radial-gradient(circle at 100% 22%, rgba(46, 125, 82, 0.10), transparent 46%),
        radial-gradient(circle at 20% 100%, rgba(201, 162, 75, 0.10), transparent 40%),
        var(--cream);
    color: var(--ink);
    font-family: var(--font-body);
}

/* subtle paper / leaf-grain texture over the page background
   (pointer-events: none so it can never block a tap or click) */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.07;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.08  0 0 0 0 0.26  0 0 0 0 0.18  0 0 0 0.9 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    position: relative;
    z-index: 1;
    padding: 5.6rem 2.5rem 3rem 2.5rem;
    max-width: 1320px;
}

.stApp p,
.stApp label,
.stApp li,
.stMarkdown,
.stCaption {
    color: var(--ink);
    font-family: var(--font-body);
}

h1, h2, h3, h4 {
    font-family: var(--font-display) !important;
    color: var(--forest) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--gold), var(--sage), var(--gold), transparent) !important;
    margin: 2.4rem 0 !important;
}

:focus-visible {
    outline: 3px solid var(--gold) !important;
    outline-offset: 2px;
}

/* =====================================================================
   SECTION HEADINGS
   ===================================================================== */
.anchor {
    display: block;
    height: 0;
    scroll-margin-top: 96px;
}

.sec-head {
    margin: 0.2rem 0 0.4rem 0;
}

.sec-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    font-family: var(--font-body);
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--leaf);
}

.sec-eyebrow::before {
    content: "";
    width: 32px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold));
}

.stApp .sec-head .sec-title {
    font-size: clamp(1.7rem, 3.1vw, 2.5rem) !important;
    line-height: 1.1 !important;
    margin: 0.3rem 0 0 0 !important;
    padding: 0 !important;
}

.stApp .sec-head h1.sec-title {
    font-size: clamp(2.1rem, 4.2vw, 3.3rem) !important;
}

/* =====================================================================
   FIXED NAVIGATION (glass pill)
   ===================================================================== */
.topnav {
    position: fixed;
    top: 14px;
    left: 50%;
    z-index: 1000;
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 6px 8px 6px 16px;
    border-radius: 999px;
    background: rgba(6, 36, 26, 0.72);
    -webkit-backdrop-filter: blur(14px) saturate(1.4);
    backdrop-filter: blur(14px) saturate(1.4);
    border: 1px solid rgba(201, 162, 75, 0.45);
    box-shadow: 0 14px 34px rgba(6, 36, 26, 0.30),
                inset 0 1px 0 rgba(255, 255, 255, 0.10);
    animation: navIn 1.1s var(--ease) both;
    pointer-events: auto;
}

.stApp .topnav a {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 14px;
    border-radius: 999px;
    color: #E8F4E2 !important;
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.2px;
    text-decoration: none !important;
    transition: background 0.3s var(--ease), color 0.3s var(--ease),
                transform 0.3s var(--ease);
}

.stApp .topnav a:hover {
    background: rgba(255, 255, 255, 0.14);
    color: #D8F8B6 !important;
    transform: translateY(-1px);
}

.stApp .topnav a.brand {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 0.95rem;
    margin-right: 6px;
    padding-left: 0;
}

.stApp .topnav a.brand:hover {
    background: transparent;
    transform: none;
}

.topnav .ic {
    font-size: 0.95rem;
}

@keyframes navIn {
    from { opacity: 0; transform: translate(-50%, -34px); }
    to   { opacity: 1; transform: translate(-50%, 0); }
}

@media (max-width: 760px) {
    .block-container { padding: 5.2rem 1rem 2rem 1rem; }
    .topnav { padding: 5px 6px 5px 12px; }
    .stApp .topnav a { padding: 8px 10px; }
    .topnav .tx { display: none; }
    .stApp .topnav a.brand .tx { display: inline; }
}

/* =====================================================================
   AMBIENT FLOATING LEAVES (behind all content)
   FIX: pointer-events none on the whole layer and everything inside it.
   ===================================================================== */
.leaf-field {
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    overflow: hidden;
}

.leaf-field,
.leaf-field * {
    pointer-events: none !important;
}

.leaf {
    position: absolute;
    width: var(--s);
    height: var(--s);
    color: var(--c);
    pointer-events: none;
}

.leaf svg {
    display: block;
    width: 100%;
    height: 100%;
    fill: currentColor;
}

.leaf-field .leaf {
    top: -10vh;
    left: var(--x);
    opacity: 0;
    will-change: transform, opacity;
    animation: leafDrift var(--d) linear var(--dl) infinite;
}

.leaf-field .leaf svg {
    filter: drop-shadow(0 5px 6px rgba(6, 36, 26, 0.16));
}

@keyframes leafDrift {
    0%   { transform: translate3d(0, -8vh, 0) rotate(0deg); opacity: 0; }
    8%   { opacity: 0.42; }
    25%  { transform: translate3d(5vw, 25vh, 0) rotate(80deg); }
    50%  { transform: translate3d(-4vw, 55vh, 0) rotate(190deg); opacity: 0.38; }
    75%  { transform: translate3d(6vw, 82vh, 0) rotate(280deg); opacity: 0.32; }
    100% { transform: translate3d(0, 114vh, 0) rotate(370deg); opacity: 0; }
}

/* =====================================================================
   CINEMATIC INTRO (plays once per session, ~5 seconds)
   FIX: the intro is a full-screen overlay. It used to keep catching every
   tap/click until its animation finished (and forever if the animation was
   paused or throttled). It now NEVER intercepts pointer events, and when it
   finishes it is also pushed behind everything (z-index -1).
   ===================================================================== */
.intro {
    position: fixed;
    inset: 0;
    z-index: 99999;
    overflow: hidden;
    pointer-events: none;
    animation: introGone 0.01s linear 4.95s forwards;
}

.intro,
.intro * {
    pointer-events: none !important;
}

@keyframes introGone {
    to { visibility: hidden; opacity: 0; z-index: -1; }
}

.intro-panel {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 50.6%;
    z-index: 0;
}

.intro-panel.left {
    left: 0;
    background: linear-gradient(90deg, #04180F, #0A3A2A);
    animation: gateL 1.2s cubic-bezier(.77, 0, .18, 1) 3.7s forwards;
}

.intro-panel.right {
    right: 0;
    background: linear-gradient(90deg, #0A3A2A, #04180F);
    animation: gateR 1.2s cubic-bezier(.77, 0, .18, 1) 3.7s forwards;
}

@keyframes gateL { to { transform: translateX(-102%); } }
@keyframes gateR { to { transform: translateX(102%); } }

.intro-seam {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 2px;
    margin-left: -1px;
    z-index: 2;
    opacity: 0;
    transform: scaleY(0);
    background: linear-gradient(transparent, var(--gold), transparent);
    animation: seamIn 1s ease 0.2s forwards, seamOut 0.4s ease 3.6s forwards;
}

@keyframes seamIn  { to { opacity: 1; transform: scaleY(1); } }
@keyframes seamOut { to { opacity: 0; } }

.intro-stage {
    position: absolute;
    inset: 0;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: stageOut 0.7s ease 3.5s forwards;
}

@keyframes stageOut {
    to { opacity: 0; transform: scale(1.1); filter: blur(8px); }
}

.intro-glow {
    position: absolute;
    width: 80vmin;
    height: 80vmin;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(168, 230, 192, 0.55), rgba(46, 125, 82, 0.25) 40%, transparent 70%);
    filter: blur(12px);
    opacity: 0;
    transform: scale(0.4);
    animation: glowIn 2.2s var(--ease) 0.1s forwards,
               glowPulse 2.6s ease-in-out 2.3s infinite;
}

@keyframes glowIn   { to { opacity: 1; transform: scale(1); } }
@keyframes glowPulse {
    0%, 100% { filter: blur(12px) brightness(1); }
    50%      { filter: blur(16px) brightness(1.25); }
}

.intro-rings span {
    position: absolute;
    left: 50%;
    top: 50%;
    width: 22vmin;
    height: 22vmin;
    margin: -11vmin 0 0 -11vmin;
    border-radius: 50%;
    border: 1px solid rgba(201, 162, 75, 0.6);
    opacity: 0;
    animation: ringOut 3s cubic-bezier(.2, .6, .2, 1) infinite;
}

.intro-rings span:nth-child(2) { animation-delay: 0.9s; }
.intro-rings span:nth-child(3) { animation-delay: 1.8s; }

@keyframes ringOut {
    0%   { transform: scale(0.4); opacity: 0.9; }
    100% { transform: scale(3.6); opacity: 0; }
}

.intro-leaves {
    position: absolute;
    inset: 0;
    overflow: hidden;
}

.intro-leaves .leaf {
    bottom: -14%;
    left: var(--x);
    opacity: 0;
    animation: leafRise var(--d) ease-out var(--dl) both;
}

.intro-leaves .leaf svg {
    filter: drop-shadow(0 0 8px rgba(168, 230, 192, 0.75));
}

@keyframes leafRise {
    0%   { transform: translate3d(0, 0, 0) rotate(0deg); opacity: 0; }
    15%  { opacity: 0.95; }
    60%  { transform: translate3d(7vw, -70vh, 0) rotate(170deg); }
    100% { transform: translate3d(-4vw, -125vh, 0) rotate(320deg); opacity: 0; }
}

.intro-core {
    position: relative;
    z-index: 3;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 0 18px;
}

.intro-mark {
    width: clamp(76px, 12vmin, 124px);
    animation: markIn 1.3s var(--ease) 0.3s both;
}

.intro-mark .rewa-mark {
    display: block;
    width: 100%;
    height: auto;
    animation: markGlow 2.4s ease-in-out 1.6s infinite;
}

@keyframes markIn {
    from { opacity: 0; transform: scale(0.25) rotate(-140deg); }
    to   { opacity: 1; transform: none; }
}

@keyframes markGlow {
    0%, 100% { filter: drop-shadow(0 0 10px rgba(168, 230, 192, 0.5)); }
    50%      { filter: drop-shadow(0 0 30px rgba(216, 248, 182, 0.95)); }
}

.intro-word {
    margin-top: 10px;
    font-family: 'Manrope', var(--font-body);
    font-weight: 800;
    font-size: clamp(2.8rem, 10vmin, 6rem);
    line-height: 1.05;
    letter-spacing: -1px;
    color: #FFFFFF;
    perspective: 700px;
    white-space: nowrap;
}

.intro-word span {
    display: inline-block;
    opacity: 0;
    transform: translateY(34px) rotateX(-70deg);
    transform-origin: 50% 100%;
    animation: letterIn 0.9s var(--ease) forwards;
    animation-delay: calc(1.05s + var(--i) * 0.09s);
}

.intro-word span.g {
    background: linear-gradient(180deg, #E4FBC2, #9BE08A);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

@keyframes letterIn {
    to { opacity: 1; transform: none; }
}

.intro-sub {
    margin-top: 12px;
    font-family: var(--font-body);
    font-weight: 500;
    font-size: clamp(0.62rem, 1.7vmin, 0.82rem);
    letter-spacing: 0.38em;
    padding-left: 0.38em;
    text-transform: uppercase;
    color: #CFE8D2;
    opacity: 0;
    animation: fadeUp 1s var(--ease) 1.8s forwards;
}

.intro-bar {
    width: min(240px, 52vw);
    height: 2px;
    margin-top: 24px;
    border-radius: 2px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.14);
    opacity: 0;
    animation: fadeUp 0.6s ease 1.9s forwards;
}

.intro-bar i {
    display: block;
    width: 100%;
    height: 100%;
    transform-origin: left;
    transform: scaleX(0);
    background: linear-gradient(90deg, var(--gold), #D8F8B6);
    animation: fillBar 1.5s cubic-bezier(.65, 0, .35, 1) 2s forwards;
}

@keyframes fillBar { to { transform: scaleX(1); } }

@keyframes fadeUp {
    from { opacity: 0; translate: 0 14px; }
    to   { opacity: 1; translate: none; }
}

/* =====================================================================
   HERO  (deep forest, glowing orbs, parallax leaves, grain)
   ===================================================================== */
.hero {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    min-height: clamp(440px, 64vh, 620px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 56px 110px;
    margin: 0 0 34px 0;
    border-radius: 34px;
    border: 1px solid rgba(201, 162, 75, 0.55);
    background:
        radial-gradient(ellipse 60% 80% at 88% 0%, rgba(157, 184, 155, 0.55), transparent 62%),
        radial-gradient(ellipse 50% 70% at 8% 100%, rgba(78, 154, 104, 0.50), transparent 66%),
        linear-gradient(125deg, #06241A 0%, #0C4A34 46%, #2E7D52 100%);
    box-shadow: 0 34px 64px -22px rgba(6, 36, 26, 0.60),
                inset 0 1px 0 rgba(255, 255, 255, 0.16);
}

.hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: 70px;
    right: 70px;
    height: 2px;
    z-index: 4;
    pointer-events: none;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    opacity: 0.22;
    mix-blend-mode: overlay;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
}

.hero-bg {
    position: absolute;
    inset: 0;
    z-index: 0;
    pointer-events: none;
}

.hero-bg,
.hero-bg * {
    pointer-events: none !important;
}

.hero-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(8px);
}

.hero-orb.o1 {
    width: 460px;
    height: 460px;
    right: -90px;
    top: -140px;
    background: radial-gradient(circle, rgba(200, 245, 175, 0.55), transparent 66%);
    animation: orbFloat 14s ease-in-out infinite;
}

.hero-orb.o2 {
    width: 520px;
    height: 520px;
    left: -140px;
    bottom: -220px;
    background: radial-gradient(circle, rgba(120, 200, 140, 0.50), transparent 66%);
    animation: orbFloat 18s ease-in-out -6s infinite reverse;
}

@keyframes orbFloat {
    0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
    50%      { transform: translate3d(-26px, 22px, 0) scale(1.08); }
}

.hero-layer {
    position: absolute;
    inset: 0;
    pointer-events: none;
}

.hero-layer .leaf {
    left: var(--x);
    top: var(--y);
    opacity: 0.5;
    animation: leafFloat var(--d) ease-in-out var(--dl) infinite alternate;
}

.hero-layer.l1 .leaf { filter: blur(1.5px); opacity: 0.42; }

.hero-layer .leaf svg {
    filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.25));
}

@keyframes leafFloat {
    from { transform: translate3d(-10px, 8px, 0) rotate(-14deg); }
    to   { transform: translate3d(14px, -16px, 0) rotate(16deg); }
}

.fly {
    position: absolute;
    left: var(--x);
    top: var(--y);
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #E4FBC2;
    box-shadow: 0 0 10px 3px rgba(190, 240, 170, 0.7);
    opacity: 0;
    pointer-events: none;
    animation: twinkle var(--d) ease-in-out var(--dl) infinite;
}

@keyframes twinkle {
    0%, 100% { opacity: 0; transform: scale(0.6); }
    50%      { opacity: 1; transform: scale(1.25) translateY(-8px); }
}

.hero-inner {
    position: relative;
    z-index: 3;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}

/* hero content enters after the intro (or immediately if no intro) */
.hero {
    --base: 0.15s;
}

.hero.after-intro {
    --base: 3.95s;
}

.hero .hero-item {
    animation: heroIn 1.1s var(--ease) both;
    animation-delay: calc(var(--base) + var(--n, 0) * 0.12s);
}

@keyframes heroIn {
    from { opacity: 0; translate: 0 30px; filter: blur(8px); }
    to   { opacity: 1; translate: none; filter: none; }
}

.hero .rewa-mark {
    display: block;
    width: clamp(54px, 5.6vw, 74px);
    height: auto;
    margin-bottom: 6px;
    filter: drop-shadow(0 6px 14px rgba(0, 0, 0, 0.3));
}

.hero-eyebrow {
    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.34em;
    padding-left: 0.34em;
    text-transform: uppercase;
    color: var(--gold-soft);
    margin-bottom: 4px;
}

.stApp .hero .wordmark {
    font-family: 'Manrope', var(--font-body);
    font-weight: 800;
    font-size: clamp(2.6rem, 6vw, 4.6rem);
    line-height: 1.04;
    letter-spacing: -1.5px;
    margin: 0;
    color: #FFFFFF !important;
    text-shadow: 0 6px 22px rgba(0, 0, 0, 0.28);
    white-space: nowrap;
}

.stApp .hero .wordmark .g {
    background: linear-gradient(100deg, #B8F27E 20%, #F6FFD8 50%, #B8F27E 80%);
    background-size: 220% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 7s linear infinite;
}

@keyframes shimmer {
    from { background-position: 0% 0; }
    to   { background-position: -220% 0; }
}

.stApp .hero .full-name {
    font-family: 'Manrope', var(--font-body);
    font-weight: 500;
    font-size: clamp(0.92rem, 1.6vw, 1.22rem);
    letter-spacing: 0.3px;
    margin: 8px 0 0 0;
    color: #F0FAEA !important;
}

.hero .rule {
    display: flex;
    align-items: center;
    gap: 12px;
    width: min(320px, 70%);
    margin: 16px 0 14px 0;
}

.hero .rule span.line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232, 250, 220, 0.85));
}

.hero .rule span.line.r {
    background: linear-gradient(270deg, transparent, rgba(232, 250, 220, 0.85));
}

.hero .rule-leaf {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
}

.stApp .hero .tagline {
    font-family: var(--font-body);
    font-weight: 500;
    font-size: clamp(0.64rem, 0.95vw, 0.8rem);
    text-transform: uppercase;
    letter-spacing: 0.42em;
    padding-left: 0.42em;
    margin: 0;
    color: #EAF8DF !important;
}

.hero .tagline .dot {
    margin: 0 0.9em;
    color: #D2F9A0;
}

.cta-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 12px;
    margin-top: 28px;
}

.stApp .hero a.cta {
    position: relative;
    overflow: hidden;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 13px 26px;
    border-radius: 999px;
    font-family: var(--font-body);
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.3px;
    text-decoration: none !important;
    transition: transform 0.4s var(--ease), box-shadow 0.4s var(--ease),
                background 0.4s var(--ease), color 0.4s var(--ease);
}

.stApp .hero a.cta.primary {
    color: var(--deep) !important;
    background: linear-gradient(135deg, #F7F3E7, #D8F8B6);
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28), 0 0 0 0 rgba(216, 248, 182, 0.5);
    animation: ctaPulse 3.2s ease-in-out infinite;
}

.stApp .hero a.cta.ghost {
    color: #F0FAEA !important;
    border: 1px solid rgba(240, 250, 234, 0.55);
    background: rgba(255, 255, 255, 0.08);
}

.stApp .hero a.cta:hover {
    transform: translateY(-3px) scale(1.03);
}

.stApp .hero a.cta.primary:hover {
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.32), 0 0 28px rgba(216, 248, 182, 0.65);
}

.stApp .hero a.cta.ghost:hover {
    background: rgba(255, 255, 255, 0.18);
}

@keyframes ctaPulse {
    0%, 100% { box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28), 0 0 0 0 rgba(216, 248, 182, 0.45); }
    50%      { box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28), 0 0 0 12px rgba(216, 248, 182, 0); }
}

.hero-chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-top: 22px;
}

.hero-chips .chip {
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #EAF8DF;
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(236, 251, 226, 0.35);
    -webkit-backdrop-filter: blur(4px);
    backdrop-filter: blur(4px);
}

/* icon tiles (right) */
.hero .tiles {
    position: absolute;
    z-index: 3;
    right: 34px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    gap: 10px;
    pointer-events: none;
}

.hero .tile {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 15px;
    border: 1px solid rgba(236, 251, 226, 0.55);
    background: rgba(255, 255, 255, 0.10);
    -webkit-backdrop-filter: blur(4px);
    backdrop-filter: blur(4px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14);
    animation: tileBob 5s ease-in-out infinite;
    transition: background 0.3s ease, border-color 0.3s ease, scale 0.3s var(--ease);
}

.hero .tile:nth-child(2) { animation-delay: -1.2s; }
.hero .tile:nth-child(3) { animation-delay: -2.4s; }
.hero .tile:nth-child(4) { animation-delay: -3.6s; }

.hero .tile svg {
    width: 22px;
    height: 22px;
}

@keyframes tileBob {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-6px); }
}

/* Optional: user-supplied banner picture */
.hero.image-mode {
    padding: 0;
    min-height: 0;
    display: block;
}

.hero img.logo-img {
    display: block;
    width: 100%;
    height: auto;
}

@media (max-width: 1100px) {
    .hero { padding: 52px 80px; }
}

@media (max-width: 900px) {
    .hero .tiles { display: none; }
}

@media (max-width: 760px) {
    .hero {
        padding: 44px 18px;
        min-height: 420px;
        border-radius: 24px;
    }
    .stApp .hero .tagline { letter-spacing: 0.25em; padding-left: 0.25em; }
}

/* =====================================================================
   SCROLL-DRIVEN MOTION (parallax + reveals)
   Works in browsers that support CSS scroll timelines (Chrome, Edge,
   recent Safari). Others get a gentle fade-in instead.

   FIX: scroll-driven "reveal" animations were applied to Streamlit's
   interactive containers (bordered panels, expanders, charts). That kept
   them transformed / faded and could leave buttons, uploaders and the
   camera unresponsive. Reveals are now limited to plain, non-interactive
   markup (section headings, reward cards, metrics).
   ===================================================================== */
@keyframes heroParallax {
    to { translate: 0 var(--py, 80px); }
}

@keyframes heroFade {
    to { opacity: 0.12; translate: 0 56px; scale: 0.97; }
}

@keyframes revealUp {
    from { opacity: 0; translate: 0 42px; scale: 0.985; }
    to   { opacity: 1; translate: none; scale: none; }
}

@supports (animation-timeline: scroll()) {
    .hero-layer {
        animation: heroParallax linear both;
        animation-timeline: scroll(nearest);
        animation-range: 0px 700px;
    }
    .hero-inner {
        animation: heroFade linear both;
        animation-timeline: scroll(nearest);
        animation-range: 0px 540px;
    }
}

@supports (animation-timeline: view()) {
    .reward-card,
    .sec-head {
        animation: revealUp linear both;
        animation-timeline: view();
        animation-range: entry 0px entry 260px;
    }
}

@supports not (animation-timeline: view()) {
    .reward-card,
    .sec-head {
        animation: revealUp 0.9s var(--ease) both;
    }
}

/* =====================================================================
   BORDERED PANELS (st.container(border=True))
   FIX: no backdrop-filter and no hover-transform on these wrappers. They
   contain buttons, uploaders and the camera; a transform / blur on an
   ancestor moves the target under your finger or cursor and can break
   fixed-position popups (dropdown menus, camera permission prompt).
   ===================================================================== */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(201, 162, 75, 0.35) !important;
    border-radius: 22px !important;
    padding: 8px 10px;
    box-shadow: 0 12px 30px rgba(20, 67, 47, 0.08);
    transition: box-shadow 0.45s var(--ease), border-color 0.45s var(--ease);
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(201, 162, 75, 0.65) !important;
    box-shadow: 0 22px 46px rgba(20, 67, 47, 0.14);
}

/* AI result card */
.result {
    position: relative;
    background: linear-gradient(135deg, var(--sage-soft), #F7FBF4);
    border-radius: 18px;
    padding: 18px 22px;
    border-left: 6px solid var(--gold);
    margin: 10px 0 14px 0;
    transition: transform 0.4s var(--ease), box-shadow 0.4s var(--ease);
}

.result:hover {
    transform: translateX(4px);
    box-shadow: 0 10px 26px rgba(20, 67, 47, 0.12);
}

.result h3 {
    color: var(--forest) !important;
    margin: 0 0 4px 0 !important;
}

.result p {
    color: var(--ink) !important;
    margin: 0 !important;
}

/* ReWA AI Buddy banner */
.buddy-head {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, var(--deep), var(--forest) 60%, var(--leaf));
    border: 1px solid rgba(201, 162, 75, 0.6);
    border-radius: 18px;
    padding: 16px 22px;
    margin: 18px 0 12px 0;
}

.buddy-head::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    width: 40%;
    left: -50%;
    pointer-events: none;
    background: linear-gradient(100deg, transparent, rgba(255, 255, 255, 0.16), transparent);
    animation: sheen 6s ease-in-out infinite;
}

@keyframes sheen {
    0%, 55% { left: -50%; }
    100%    { left: 130%; }
}

.stApp .buddy-head .buddy-title {
    font-family: var(--font-display);
    color: #FFFFFF !important;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}

.stApp .buddy-head .buddy-sub {
    color: var(--gold-soft) !important;
    font-size: 0.85rem;
    letter-spacing: 0.8px;
    margin: 2px 0 0 0;
}

/* Reward cards */
.reward-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, #FFFFFF, var(--paper));
    border: 1px solid rgba(201, 162, 75, 0.35);
    border-top: 4px solid var(--gold);
    border-radius: 20px;
    padding: 20px 18px;
    text-align: center;
    box-shadow: 0 12px 26px rgba(20, 67, 47, 0.08);
    transition: transform 0.45s var(--ease), box-shadow 0.45s var(--ease),
                border-color 0.45s var(--ease);
}

.reward-card::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    width: 45%;
    left: -70%;
    pointer-events: none;
    background: linear-gradient(100deg, transparent, rgba(255, 255, 255, 0.7), transparent);
    transition: left 0.8s var(--ease);
}

.reward-card:hover {
    transform: translateY(-7px);
    border-top-color: var(--leaf);
    box-shadow: 0 24px 44px rgba(20, 67, 47, 0.18);
}

.reward-card:hover::after {
    left: 130%;
}

.reward-card .big {
    font-family: var(--font-display);
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--forest);
    line-height: 1.2;
}

.reward-card .small {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6B5B2A;
}

.badge {
    display: inline-block;
    background: var(--gold-soft);
    color: var(--forest);
    border: 1px solid var(--gold);
    border-radius: 999px;
    padding: 2px 12px;
    font-weight: 700;
    font-size: 0.85rem;
}

/* Streamlit metrics as soft cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid rgba(201, 162, 75, 0.35);
    border-radius: 18px;
    padding: 14px 18px;
    box-shadow: 0 8px 18px rgba(20, 67, 47, 0.06);
    transition: transform 0.4s var(--ease), box-shadow 0.4s var(--ease);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 30px rgba(20, 67, 47, 0.13);
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display);
    color: var(--forest);
    font-weight: 800;
}

/* =====================================================================
   BUTTONS
   ===================================================================== */
div.stButton > button {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, var(--forest), var(--emerald));
    color: #FFFFFF !important;
    border-radius: 14px;
    border: 1px solid var(--gold);
    min-height: 46px;
    font-family: var(--font-body);
    font-weight: 600;
    letter-spacing: 0.3px;
    box-shadow: 0 8px 18px rgba(6, 36, 26, 0.24);
    touch-action: manipulation;
    transition: transform 0.35s var(--ease), box-shadow 0.35s var(--ease),
                background 0.35s var(--ease);
}

div.stButton > button::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(110deg, transparent 30%, rgba(255, 255, 255, 0.30) 50%, transparent 70%);
    transform: translateX(-120%);
    transition: transform 0.8s var(--ease);
    pointer-events: none;
}

div.stButton > button p {
    color: #FFFFFF !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, var(--leaf), var(--forest));
    border-color: var(--gold);
    color: #FFFFFF !important;
    transform: translateY(-3px);
    box-shadow: 0 16px 30px rgba(46, 125, 82, 0.38);
}

div.stButton > button:hover::after {
    transform: translateX(120%);
}

div.stButton > button:active {
    transform: translateY(0) scale(0.985);
    box-shadow: 0 6px 12px rgba(6, 36, 26, 0.22);
}

div.stButton > button:disabled {
    background: #B9C7BE;
    border-color: #B9C7BE;
    box-shadow: none;
    color: #FFFFFF !important;
    transform: none;
}

div.stButton > button:disabled::after {
    display: none;
}

/* Danger buttons (type="primary"): used for destructive confirmations */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #B93B3B, #8E2323);
    border: 1px solid #8E2323;
    box-shadow: 0 8px 18px rgba(142, 35, 35, 0.30);
}

div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #D64545, #A52A2A);
    border-color: #A52A2A;
    box-shadow: 0 16px 28px rgba(165, 42, 42, 0.38);
}

.reset-panel {
    background: linear-gradient(135deg, #FFF6F4, #FFFFFF);
    border: 1px solid rgba(185, 59, 59, 0.35);
    border-left: 6px solid #B93B3B;
    border-radius: 18px;
    padding: 16px 20px;
    margin: 10px 0 12px 0;
}

.reset-panel .reset-title {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 700;
    color: #8E2323;
    margin: 0 0 4px 0;
}

.reset-panel .reset-text {
    font-size: 0.92rem;
    color: var(--ink);
    margin: 0;
}

/* On touch screens there is no real hover: don't leave buttons "stuck"
   lifted after a tap. */
@media (hover: none) {
    div.stButton > button:hover {
        transform: none;
    }
    .reward-card:hover,
    [data-testid="stMetric"]:hover {
        transform: none;
    }
}

/* =====================================================================
   INPUTS: tabs, uploader, camera, expanders, alerts
   ===================================================================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid rgba(201, 162, 75, 0.45);
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: var(--forest);
    border-radius: 12px 12px 0 0;
    padding: 8px 16px;
    transition: background 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(157, 184, 155, 0.25);
}

.stTabs [aria-selected="true"] {
    background: var(--sage-soft);
    color: var(--forest);
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--gold);
}

[data-testid="stCameraInput"] {
    border-radius: 18px;
    overflow: hidden;
}

[data-testid="stCameraInput"] button {
    border-radius: 12px;
}

[data-testid="stFileUploader"] {
    border-radius: 16px;
}

[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #FFFFFF, var(--mist));
    border: 2px dashed rgba(46, 125, 82, 0.45);
    border-radius: 18px;
    transition: border-color 0.35s ease, box-shadow 0.35s ease,
                background 0.35s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold);
    background: linear-gradient(135deg, #FFFFFF, var(--sage-soft));
    box-shadow: 0 0 0 6px rgba(201, 162, 75, 0.14);
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small {
    color: var(--ink) !important;
}

.batch-pill {
    display: inline-block;
    background: var(--gold-soft);
    color: var(--forest);
    border: 1px solid var(--gold);
    border-radius: 999px;
    padding: 3px 14px;
    font-weight: 700;
    font-size: 0.85rem;
    margin: 4px 0 8px 0;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(201, 162, 75, 0.35) !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, 0.72);
    margin-bottom: 12px;
    transition: box-shadow 0.4s var(--ease), border-color 0.4s var(--ease);
}

[data-testid="stExpander"]:hover {
    border-color: rgba(201, 162, 75, 0.7) !important;
    box-shadow: 0 14px 30px rgba(20, 67, 47, 0.10);
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

[data-testid="stImage"] img {
    border-radius: 18px;
    box-shadow: 0 10px 24px rgba(11, 46, 34, 0.18);
    transition: transform 0.5s var(--ease), box-shadow 0.5s var(--ease);
}

[data-testid="stImage"] img:hover {
    transform: scale(1.025);
    box-shadow: 0 18px 36px rgba(11, 46, 34, 0.26);
}

/* =====================================================================
   PROGRESS BAR (daily goal)
   ===================================================================== */
.pbar-wrap {
    margin: 6px 0 14px 0;
}

.pbar-label {
    font-size: 0.95rem;
    color: var(--ink);
    margin-bottom: 6px;
}

.pbar-track {
    width: 100%;
    height: 12px;
    background: rgba(20, 67, 47, 0.07);
    border: 1px solid rgba(20, 67, 47, 0.10);
    border-radius: 999px;
    overflow: hidden;
}

.pbar-fill {
    position: relative;
    height: 100%;
    background: linear-gradient(90deg, var(--forest), var(--fern));
    border-radius: 999px;
    transform-origin: left;
    animation: barGrow 1.3s var(--ease) both;
    transition: width 0.5s var(--ease);
}

.pbar-fill::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(100deg, transparent 30%, rgba(255, 255, 255, 0.45) 50%, transparent 70%);
    background-size: 200% 100%;
    animation: barSheen 3s linear infinite;
}

@keyframes barGrow  { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes barSheen { from { background-position: 200% 0; } to { background-position: -100% 0; } }

/* =====================================================================
   VIRTUAL BIN CARDS (accent colour per bin via CSS variables)
   ===================================================================== */
.bin-card {
    padding: 4px 4px 2px 4px;
}

.bin-head {
    display: flex;
    align-items: center;
    gap: 14px;
    background: linear-gradient(135deg, var(--tint), rgba(255, 255, 255, 0));
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 16px;
}

.bin-medal {
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
    box-shadow: 0 0 0 4px var(--tint), 0 6px 14px rgba(11, 46, 34, 0.12);
    transition: transform 0.5s var(--ease);
}

.bin-card:hover .bin-medal {
    transform: rotate(-10deg) scale(1.1);
}

.bin-name {
    font-family: var(--font-display);
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--forest);
    line-height: 1.1;
}

.bin-id {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6B5B2A;
    margin-top: 5px;
}

.bin-count-row {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin: 20px 4px 16px 4px;
}

.bin-count {
    font-family: var(--font-display);
    font-size: 2.9rem;
    font-weight: 800;
    color: var(--ink);
    line-height: 1;
}

.bin-count-label {
    font-size: 0.74rem;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #5B6B60;
}

.bin-cap-row {
    display: flex;
    justify-content: space-between;
    margin: 0 4px 7px 4px;
    font-size: 0.74rem;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #5B6B60;
}

.bin-cap-val {
    font-weight: 700;
    color: var(--ink);
}

.bin-track {
    height: 10px;
    margin: 0 4px;
    background: rgba(20, 67, 47, 0.07);
    border: 1px solid rgba(20, 67, 47, 0.10);
    border-radius: 999px;
    overflow: hidden;
}

.bin-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--soft));
    transform-origin: left;
    animation: barGrow 1.2s var(--ease) both;
    transition: width 0.5s var(--ease);
}

.bin-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin: 18px 4px 10px 4px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.bin-status .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}

.bin-status.ok   { background: #E3F3E8; color: #1F6F49; }
.bin-status.warn { background: #FBF0CF; color: #8A6A12; }
.bin-status.full { background: #FBE1E1; color: #9B2C2C; }

.bin-status.ok .dot   { animation: statusPulse 2.4s ease-in-out infinite; }
.bin-status.warn .dot { animation: statusPulse 1.6s ease-in-out infinite; }
.bin-status.full .dot { animation: statusPulse 0.9s ease-in-out infinite; }

@keyframes statusPulse {
    0%, 100% { box-shadow: 0 0 0 0 currentColor; opacity: 1; }
    50%      { box-shadow: 0 0 0 5px transparent; opacity: 0.55; }
}

/* =====================================================================
   FOOTER
   ===================================================================== */
.footer {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 38px 26px;
    margin-top: 34px;
    border-radius: 28px;
    border: 1px solid rgba(201, 162, 75, 0.5);
    background:
        radial-gradient(ellipse 60% 100% at 85% 0%, rgba(157, 184, 155, 0.35), transparent 60%),
        linear-gradient(120deg, var(--deep), var(--emerald));
    font-family: var(--font-display);
    letter-spacing: 1px;
}

.stApp .footer,
.stApp .footer * {
    color: #E4F1DF !important;
}

.footer .footer-leaf {
    display: inline-block;
    font-size: 1.6rem;
    margin-bottom: 6px;
    animation: tileBob 4s ease-in-out infinite;
}

.footer .footer-sub {
    font-family: var(--font-body);
    font-size: 0.9rem;
    letter-spacing: 0.4px;
    opacity: 0.85;
}

/* =====================================================================
   ACCESSIBILITY: respect "reduce motion"
   ===================================================================== */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
    }
    .intro,
    .leaf-field {
        display: none !important;
    }
    /* with animations off, hero items must still be visible */
    .hero .hero-item {
        opacity: 1 !important;
        translate: none !important;
        filter: none !important;
    }
}
</style>
"""

NAV_HTML = (
    '<div class="topnav" role="navigation" aria-label="Page sections">'
    '<a class="brand" href="#top"><span class="ic">♻️</span><span class="tx">ReWA.AI</span></a>'
    '<a href="#upload"><span class="ic">📷</span><span class="tx">Classify</span></a>'
    '<a href="#rewards"><span class="ic">🏆</span><span class="tx">Rewards</span></a>'
    '<a href="#station"><span class="ic">♻️</span><span class="tx">Station</span></a>'
    '<a href="#analytics"><span class="ic">📊</span><span class="tx">Analytics</span></a>'
    '<a href="#bins"><span class="ic">🏭</span><span class="tx">Bins</span></a>'
    '</div>'
)

AMBIENT_LEAVES_HTML = (
    '<div class="leaf-field" aria-hidden="true">'
    + leaf_field(14, seed=11, size=(18, 46), duration=(20, 38), delay=(-34, 0))
    + '</div>'
)


def build_intro_html():
    """The one-time cinematic opening (pure CSS, auto-dismisses itself)."""
    letters = "".join(
        f'<span class="{"g" if i >= 3 else ""}" style="--i:{i}">{ch}</span>'
        for i, ch in enumerate("ReWA.AI")
    )
    return (
        '<div class="intro" role="presentation" aria-hidden="true">'
        '<div class="intro-panel left"></div>'
        '<div class="intro-panel right"></div>'
        '<div class="intro-seam"></div>'
        '<div class="intro-stage">'
        '<div class="intro-glow"></div>'
        '<div class="intro-rings"><span></span><span></span><span></span></div>'
        '<div class="intro-leaves">'
        + leaf_field(
            16, seed=5, size=(16, 40), duration=(3.4, 5.0),
            delay=(0.0, 1.9), colors=INTRO_LEAF_COLORS
        )
        + fireflies(14, seed=3)
        + '</div>'
        '<div class="intro-core">'
        f'<div class="intro-mark">{INTRO_MARK_SVG}</div>'
        f'<div class="intro-word">{letters}</div>'
        '<div class="intro-sub">Recycling-Reusable Waste Assistant</div>'
        '<div class="intro-bar"><i></i></div>'
        '</div>'
        '</div>'
        '</div>'
    )


def build_hero_html(after_intro, bin_count):
    """The hero section: parallax leaves, glowing orbs, wordmark and CTAs."""
    if USE_BANNER_IMAGE:
        banner_mime, banner_b64 = load_banner_base64()
        if banner_b64:
            return (
                '<div class="hero image-mode" id="top">'
                f'<img class="logo-img" src="data:{banner_mime};base64,{banner_b64}" '
                'alt="ReWA.AI - Recycling-Reusable Waste Assistant">'
                '</div>'
            )

    hero_class = "hero after-intro" if after_intro else "hero"

    # NOTE: the HTML is deliberately written without indentation or blank
    # lines; Streamlit's markdown would otherwise treat it as a code block.
    return (
        f'<div class="{hero_class}" id="top">'
        '<div class="hero-bg" aria-hidden="true">'
        '<div class="hero-orb o1"></div>'
        '<div class="hero-orb o2"></div>'
        '<div class="hero-layer l1" style="--py:70px">'
        + leaf_field(6, seed=21, size=(46, 86), duration=(7, 12),
                     delay=(-8, 0), colors=HERO_LEAF_COLORS, spread_y=True)
        + '</div>'
        '<div class="hero-layer l2" style="--py:150px">'
        + leaf_field(8, seed=33, size=(20, 40), duration=(5, 9),
                     delay=(-6, 0), colors=HERO_LEAF_COLORS, spread_y=True)
        + fireflies(16, seed=8)
        + '</div>'
        '</div>'
        f'{TILES_HTML}'
        '<div class="hero-inner">'
        f'<div class="hero-item" style="--n:0">{LOGO_MARK_SVG}</div>'
        '<div class="hero-item hero-eyebrow" style="--n:1">Recycle • Reuse • Reward</div>'
        '<div class="hero-item wordmark" style="--n:2">ReW<span class="g">A.AI</span></div>'
        '<div class="hero-item full-name" style="--n:3">(Recycling-Reusable Waste Assistant)</div>'
        '<div class="hero-item rule" style="--n:4">'
        '<span class="line"></span>'
        f'{DIVIDER_LEAF_SVG}'
        '<span class="line r"></span>'
        '</div>'
        '<div class="hero-item tagline" style="--n:5">Sort Today<span class="dot">•</span>Save Tomorrow</div>'
        '<div class="hero-item cta-row" style="--n:6">'
        '<a class="cta primary" href="#upload">📷 Start sorting</a>'
        '<a class="cta ghost" href="#rewards">🏆 See your impact</a>'
        '</div>'
        '<div class="hero-item hero-chips" style="--n:7">'
        '<span class="chip">🧠 CLIP vision AI</span>'
        f'<span class="chip">🗑️ {bin_count} smart bins</span>'
        f'<span class="chip">🌟 {POINTS_PER_EVENT} Eco Points per item</span>'
        '</div>'
        '</div>'
        '</div>'
    )


def section_head(eyebrow, title, anchor=None, tag="h2"):
    """A designed section heading with a small eyebrow label."""
    anchor_html = f'<div id="{anchor}" class="anchor"></div>' if anchor else ""
    st.markdown(
        f'{anchor_html}'
        '<div class="sec-head">'
        f'<div class="sec-eyebrow">{eyebrow}</div>'
        f'<{tag} class="sec-title">{title}</{tag}>'
        '</div>',
        unsafe_allow_html=True
    )


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


# ---------- OPENING ANIMATION + THEME + NAV + HERO ----------
# The intro lives in an st.empty() slot so the page layout never shifts
# between the first run and later reruns.
intro_slot = st.empty()
if show_intro:
    intro_slot.markdown(build_intro_html(), unsafe_allow_html=True)

# NOTE: all HTML below is written without indentation or blank lines
# (outside the <style> block); Streamlit's markdown would otherwise treat
# it as a code block.
st.markdown(
    THEME_CSS
    + "\n"
    + NAV_HTML
    + AMBIENT_LEAVES_HTML
    + build_hero_html(
        hero_after_intro,
        len(st.session_state.digital_twin_bins)
    ),
    unsafe_allow_html=True
)

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

section_head("Your impact", "🏆 Your Eco Rewards", anchor="rewards")

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

section_head("Get started", "Welcome to a greener future!", anchor="upload", tag="h1")
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
section_head("Sorting", "♻️ Virtual Recycling Station", anchor="station")

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
section_head("Insights", "📊 Recycling Analytics", anchor="analytics")

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
section_head("Activity", "🕘 Recent Recycling Events")

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
section_head("Digital twin", "🏭 Your Recycling Bins", anchor="bins")
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
<div class="footer-leaf">🌱</div>
<div>ReWA.AI &nbsp;•&nbsp; Reduce waste • Reuse • Recycle</div>
<div class="footer-sub">Right item. Right bin. A cleaner tomorrow.</div>
</div>
""", unsafe_allow_html=True) 