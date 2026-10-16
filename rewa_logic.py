"""Pure helper logic for ReWA.AI.

No Streamlit imports here on purpose, so everything can be unit-tested
with plain `pytest` (see test_rewa_logic.py).

Everything in this file is DERIVED from real saved data (the rewards
summary and the recycling event log). Nothing here stores or invents
numbers, so badges, garden, missions and points can never disagree with
each other: they all read the same summary.
"""
import io

# ---------------------------------------------------------------------------
# 1. Recording an item and awarding its points (exactly once)
# ---------------------------------------------------------------------------


def _award(uid, label, category, pending_rewards, award_fn):
    """Try to award points. Remember the event for a retry if saving fails."""
    try:
        awarded = award_fn(uid, label, category)
    except Exception as error:  # database locked, file missing, ...
        pending_rewards[uid] = (label, category)
        return "points_err", str(error)
    pending_rewards.pop(uid, None)
    return ("ok" if awarded else "rewarded"), ""


def record_and_reward(uid, label, category, *, recorded_uids, recorded_bins,
                      pending_rewards, record_fn, award_fn,
                      fingerprint=None, recorded_fingerprints=None):
    """Record one confirmed item, then award points for it.

    Status values:
      ok           recorded and rewarded
      rewarded     recorded, database says this event was already rewarded
      points_err   recorded, but points could not be saved (will be retried)
      duplicate    this uid was already recorded -> nothing happens
      same_photo   this exact photo was already recorded this session
      full         the bin refused the item -> nothing recorded, no points
      record_err   recording crashed -> nothing recorded, no points
    """
    if uid in recorded_uids:
        return "duplicate", ""

    if (fingerprint and recorded_fingerprints is not None
            and fingerprint in recorded_fingerprints):
        return "same_photo", ""

    try:
        recorded = record_fn(category, label)
    except Exception as error:
        return "record_err", str(error)

    if not recorded:
        return "full", ""

    recorded_uids.add(uid)
    recorded_bins[uid] = category
    if fingerprint and recorded_fingerprints is not None:
        recorded_fingerprints.add(fingerprint)
    return _award(uid, label, category, pending_rewards, award_fn)


def retry_pending_rewards(pending_rewards, award_fn):
    """Retry points that failed to save earlier. Returns how many succeeded.

    award_fn must be idempotent per uid (the rewards database already
    refuses to reward the same event_key twice), so retrying is safe.
    """
    recovered = 0
    for uid, (label, category) in list(pending_rewards.items()):
        status, _ = _award(uid, label, category, pending_rewards, award_fn)
        if status in ("ok", "rewarded"):
            recovered += 1
    return recovered


def reward_gap(recorded_items, rewarded_events):
    """How many recorded items have no reward entry (never negative)."""
    return max(int(recorded_items) - int(rewarded_events), 0)


# ---------------------------------------------------------------------------
# 2. Classification uncertainty
# ---------------------------------------------------------------------------


def needs_manual_category(score, suggested_bin, threshold=0.55):
    """True when the user must pick the bin themselves."""
    return suggested_bin is None or score < threshold


# ---------------------------------------------------------------------------
# 3. Image validation
# ---------------------------------------------------------------------------

ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
MAX_PIXELS = 40_000_000


def load_validated_image(data, max_bytes=8 * 1024 * 1024, max_side=1600):
    """Return a safe, RGB, size-limited PIL image or raise ValueError.

    The ValueError message is written for end users, so it can be shown.
    """
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise ValueError("The file is empty.")
    if len(data) > max_bytes:
        raise ValueError(f"The file is larger than {max_bytes // (1024 * 1024)} MB.")

    try:
        Image.open(io.BytesIO(data)).verify()  # cheap integrity check
        image = Image.open(io.BytesIO(data))  # verify() invalidates the object
        if image.format not in ALLOWED_FORMATS:
            raise ValueError("Only JPG, PNG and WEBP images are supported.")
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("The image has too many pixels. Please use a smaller photo.")
        image = image.convert("RGB")
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError("This file is not a readable image.") from error

    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))  # faster model, same photo
    return image


# ---------------------------------------------------------------------------
# 4. Rewards extras: badges, garden, missions, passport (all derived)
# ---------------------------------------------------------------------------

BADGES = [
    # key, icon, title, description, metric, target
    ("first", "🌱", "First Sprout", "Record your first item", "events", 1),
    ("five", "🌿", "Getting Growing", "Record 5 items", "events", 5),
    ("twentyfive", "🌳", "Eco Regular", "Record 25 items", "events", 25),
    ("streak3", "🔥", "3-Day Streak", "Record items on 3 days in a row", "streak", 3),
    ("streak7", "🏆", "7-Day Streak", "Record items on 7 days in a row", "streak", 7),
    ("explorer", "🧭", "Sorting Explorer", "Earn points in 3 different bins", "bins", 3),
    ("hazard", "☣️", "Hazard Aware", "Record one hazardous item", "hazard", 1),
]


def reward_metrics(summary, points_rows):
    rows = points_rows or []
    earned_bins = {r.get("category") for r in rows if (r.get("points") or 0) > 0}
    return {
        "events": int(summary.get("total_events", 0) or 0),
        "streak": int(summary.get("longest_streak", 0) or 0),
        "bins": len(earned_bins),
        "hazard": 1 if "Hazardous" in earned_bins else 0,
    }


def badge_progress(summary, points_rows):
    """List of badges with earned flag and current/target progress."""
    metrics = reward_metrics(summary, points_rows)
    result = []
    for key, icon, title, desc, metric, target in BADGES:
        current = metrics[metric]
        result.append({
            "key": key, "icon": icon, "title": title, "description": desc,
            "current": min(current, target), "target": target,
            "earned": current >= target,
        })
    return result


GARDEN_STAGES = [
    (0, "Empty plot", "🟫"),
    (1, "Seed", "🌱"),
    (5, "Sprout", "🌿"),
    (15, "Sapling", "🪴"),
    (30, "Young tree", "🌳"),
    (60, "Grove", "🌲🌳🌲"),
]


def garden_stage(total_events):
    """Garden stage from the number of recorded (rewarded) events."""
    total_events = max(int(total_events or 0), 0)
    index = 0
    for i, (threshold, _, _) in enumerate(GARDEN_STAGES):
        if total_events >= threshold:
            index = i
    threshold, name, emoji = GARDEN_STAGES[index]
    if index + 1 < len(GARDEN_STAGES):
        next_threshold, next_name, _ = GARDEN_STAGES[index + 1]
        span = next_threshold - threshold
        progress = (total_events - threshold) / span
        remaining = next_threshold - total_events
    else:
        next_name, remaining, progress = None, 0, 1.0
    return {
        "name": name, "emoji": emoji, "next_name": next_name,
        "remaining": remaining, "progress": max(0.0, min(progress, 1.0)),
    }


def daily_missions(today_events, daily_goal, bins_used_this_session, advice_this_session):
    """Three achievable missions. Scope is stated so nothing is overclaimed.

    Mission 1 uses the saved database (today). Missions 2 and 3 only count
    what happened in this browser session.
    """
    today_events = int(today_events or 0)
    return [
        {"title": f"Record {daily_goal} items today", "scope": "saved today",
         "current": min(today_events, daily_goal), "target": daily_goal,
         "done": today_events >= daily_goal},
        {"title": "Sort items into 2 different bins", "scope": "this session",
         "current": min(bins_used_this_session, 2), "target": 2,
         "done": bins_used_this_session >= 2},
        {"title": "Read ReWA Buddy advice for 1 item", "scope": "this session",
         "current": min(advice_this_session, 1), "target": 1,
         "done": advice_this_session >= 1},
    ]


def passport_stamps(points_rows, bin_names):
    """One stamp per bin; stamped once that bin has earned points."""
    stamped = {r.get("category") for r in (points_rows or []) if (r.get("points") or 0) > 0}
    return [{"bin": name, "stamped": name in stamped} for name in bin_names]