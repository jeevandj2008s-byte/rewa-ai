"""
rewards.py - Eco Points, daily activity and streaks for EcoSortAI.

Design notes
------------
* Uses the SAME SQLite file as data_base.py, but only adds its own two tables.
  Your existing tables, Gemini code, and virtual bins are never touched.
* Points are awarded with INSERT OR IGNORE on a UNIQUE event_key, so a rerun,
  refresh, or double click can never award the same event twice.
* The reward row and the daily-activity row are written in ONE transaction:
  either both are saved or neither is.
* Streaks are computed from the saved dates, never from session state.
* Column names deliberately avoid "item_name"/"material" so that your
  "Clear recent events" button (which wipes any table having both columns)
  does not delete earned points.
"""

import sqlite3
from datetime import date, timedelta

import data_base

POINTS_PER_EVENT = 10
DAILY_GOAL = 5  # recycled items per day that count as a full daily goal


def db_path():
    """Locate the same database file used by data_base.py."""
    return (
        getattr(data_base, "DB_NAME", None)
        or getattr(data_base, "DB_PATH", None)
        or getattr(data_base, "DATABASE", None)
        or "recycling.db"
    )


def _connect():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_rewards_db():
    """Create the reward tables if they do not exist yet (safe to call every run)."""
    conn = _connect()
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reward_events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key     TEXT NOT NULL UNIQUE,
                    item_label    TEXT NOT NULL,
                    bin_name      TEXT NOT NULL,
                    points        INTEGER NOT NULL,
                    activity_date TEXT NOT NULL,
                    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_activity (
                    activity_date TEXT PRIMARY KEY,
                    events        INTEGER NOT NULL DEFAULT 0,
                    points        INTEGER NOT NULL DEFAULT 0
                )
                """
            )
    finally:
        conn.close()


def award_event(event_key, item_label, bin_name, today=None):
    """
    Award POINTS_PER_EVENT for one successfully recorded recycling event.

    Returns True if points were awarded, False if this event_key was already
    rewarded (duplicate). Raises on real database errors.
    """
    today = (today or date.today()).isoformat()
    conn = _connect()
    try:
        with conn:  # one atomic transaction (commit on success, rollback on error)
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO reward_events
                    (event_key, item_label, bin_name, points, activity_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_key, item_label, bin_name, POINTS_PER_EVENT, today),
            )
            if cur.rowcount == 0:
                return False  # already rewarded

            conn.execute(
                """
                INSERT INTO daily_activity (activity_date, events, points)
                VALUES (?, 1, ?)
                ON CONFLICT(activity_date) DO UPDATE SET
                    events = events + 1,
                    points = points + excluded.points
                """,
                (today, POINTS_PER_EVENT),
            )
        return True
    finally:
        conn.close()


def _calculate_streaks(day_strings, today):
    """Return (current_streak, longest_streak) from a list of ISO date strings."""
    days = sorted({date.fromisoformat(d) for d in day_strings})
    if not days:
        return 0, 0

    longest = run = 0
    previous = None
    for d in days:
        run = run + 1 if previous and (d - previous).days == 1 else 1
        longest = max(longest, run)
        previous = d

    # `run` is now the streak ending on the most recent active day.
    # It is still alive if that day is today or yesterday.
    current = run if (today - days[-1]).days <= 1 else 0
    return current, longest


def get_summary(today=None):
    """Everything the rewards dashboard needs, in one dict."""
    today = today or date.today()
    conn = _connect()
    try:
        totals = conn.execute(
            "SELECT COALESCE(SUM(points), 0) AS pts, COUNT(*) AS evts FROM reward_events"
        ).fetchone()
        day_rows = conn.execute(
            "SELECT activity_date FROM daily_activity ORDER BY activity_date"
        ).fetchall()
        today_row = conn.execute(
            "SELECT events, points FROM daily_activity WHERE activity_date = ?",
            (today.isoformat(),),
        ).fetchone()
    finally:
        conn.close()

    current, longest = _calculate_streaks([r["activity_date"] for r in day_rows], today)
    return {
        "total_points": totals["pts"],
        "total_events": totals["evts"],
        "active_days": len(day_rows),
        "today_events": today_row["events"] if today_row else 0,
        "today_points": today_row["points"] if today_row else 0,
        "current_streak": current,
        "longest_streak": longest,
    }


def get_points_by_category():
    """List of dicts: category, events, points (highest points first)."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT bin_name AS category, COUNT(*) AS events, SUM(points) AS points
            FROM reward_events
            GROUP BY bin_name
            ORDER BY points DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_daily_history(days=7, today=None):
    """Last `days` days (oldest first), with zeros for days without activity."""
    today = today or date.today()
    start = today - timedelta(days=days - 1)
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT activity_date, events, points FROM daily_activity
            WHERE activity_date >= ?
            """,
            (start.isoformat(),),
        ).fetchall()
    finally:
        conn.close()

    lookup = {r["activity_date"]: r for r in rows}
    history = []
    for offset in range(days):
        d = start + timedelta(days=offset)
        row = lookup.get(d.isoformat())
        history.append(
            {
                "date": d.strftime("%a %d %b"),
                "events": row["events"] if row else 0,
                "points": row["points"] if row else 0,
            }
        )
    return history