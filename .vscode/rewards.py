
"""
rewards.py - Eco Points, daily activity and streaks for EcoSortAI.

Per-user version: every row now belongs to a user_id (set by
data_base.set_current_user() at the top of each script run).

Design notes
------------
* Uses the SAME SQLite file as data_base.py, but only adds its own two tables.
* Points are awarded with INSERT OR IGNORE on a UNIQUE event_key, so a rerun,
  refresh, or double click can never award the same event twice.
* The reward row and the daily-activity row are written in ONE transaction.
* Streaks are computed from the saved dates, never from session state.
* Column names deliberately avoid "item_name"/"material" so that the
  "Clear recent events" button does not delete earned points.
* Existing databases are migrated automatically; old rows are kept under
  the user id "legacy" until claimed with claim_legacy_rewards().
"""

import sqlite3
from datetime import date, timedelta

import data_base

POINTS_PER_EVENT = 10
DAILY_GOAL = 5  # recycled items per day that count as a full daily goal
LEGACY_USER = data_base.LEGACY_USER


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


def _uid():
    # Raises if nobody is signed in, so data can never leak between users.
    return data_base._uid()


def _columns(conn, table):
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def init_rewards_db():
    """Create / migrate the reward tables (safe to call every run)."""
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
            if "user_id" not in _columns(conn, "reward_events"):
                conn.execute(
                    "ALTER TABLE reward_events "
                    f"ADD COLUMN user_id TEXT NOT NULL DEFAULT '{LEGACY_USER}'"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reward_user "
                "ON reward_events(user_id)"
            )

            # daily_activity used to be keyed by date alone. SQLite cannot
            # change a primary key, so an old table is rebuilt once.
            existing = _columns(conn, "daily_activity")
            if existing and "user_id" not in existing:
                conn.execute("ALTER TABLE daily_activity RENAME TO daily_activity_old")
                existing = []
            if not existing:
                conn.execute(
                    """
                    CREATE TABLE daily_activity (
                        user_id       TEXT NOT NULL,
                        activity_date TEXT NOT NULL,
                        events        INTEGER NOT NULL DEFAULT 0,
                        points        INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (user_id, activity_date)
                    )
                    """
                )
            has_old = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'daily_activity_old'"
            ).fetchone()
            if has_old:
                conn.execute(
                    f"""
                    INSERT INTO daily_activity (user_id, activity_date, events, points)
                    SELECT '{LEGACY_USER}', activity_date, events, points
                    FROM daily_activity_old
                    """
                )
                conn.execute("DROP TABLE daily_activity_old")
    finally:
        conn.close()


def award_event(event_key, item_label, bin_name, today=None):
    """
    Award POINTS_PER_EVENT for one successfully recorded recycling event.

    Returns True if points were awarded, False if this event_key was already
    rewarded (duplicate). Raises on real database errors.
    """
    uid = _uid()
    today = (today or date.today()).isoformat()
    conn = _connect()
    try:
        with conn:  # one atomic transaction
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO reward_events
                    (event_key, item_label, bin_name, points, activity_date, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_key, item_label, bin_name, POINTS_PER_EVENT, today, uid),
            )
            if cur.rowcount == 0:
                return False  # already rewarded

            conn.execute(
                """
                INSERT INTO daily_activity (user_id, activity_date, events, points)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(user_id, activity_date) DO UPDATE SET
                    events = events + 1,
                    points = points + excluded.points
                """,
                (uid, today, POINTS_PER_EVENT),
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

    current = run if (today - days[-1]).days <= 1 else 0
    return current, longest


def get_summary(today=None):
    """Everything the rewards dashboard needs, in one dict."""
    uid = _uid()
    today = today or date.today()
    conn = _connect()
    try:
        totals = conn.execute(
            "SELECT COALESCE(SUM(points), 0) AS pts, COUNT(*) AS evts "
            "FROM reward_events WHERE user_id = ?",
            (uid,),
        ).fetchone()
        day_rows = conn.execute(
            "SELECT activity_date FROM daily_activity "
            "WHERE user_id = ? ORDER BY activity_date",
            (uid,),
        ).fetchall()
        today_row = conn.execute(
            "SELECT events, points FROM daily_activity "
            "WHERE user_id = ? AND activity_date = ?",
            (uid, today.isoformat()),
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
            WHERE user_id = ?
            GROUP BY bin_name
            ORDER BY points DESC
            """,
            (_uid(),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_daily_history(days=7, today=None):
    """Last `days` days (oldest first), with zeros for days without activity."""
    uid = _uid()
    today = today or date.today()
    start = today - timedelta(days=days - 1)
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT activity_date, events, points FROM daily_activity
            WHERE user_id = ? AND activity_date >= ?
            """,
            (uid, start.isoformat()),
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


def reset_rewards():
    """Delete ONLY the signed-in user's points, streaks and rewarded events.

    app.py's clear_all_rewards() looks for a function with this name first,
    so its "empty every table" fallback is never used.
    """
    uid = _uid()
    conn = _connect()
    try:
        with conn:
            conn.execute("DELETE FROM reward_events WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM daily_activity WHERE user_id = ?", (uid,))
    finally:
        conn.close()


# ---------- one-time adoption of pre-login rewards ----------
def count_legacy_rewards():
    conn = _connect()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM reward_events WHERE user_id = ?", (LEGACY_USER,)
        ).fetchone()[0]
    finally:
        conn.close()


def claim_legacy_rewards(user_id):
    """Move all pre-login points and daily activity to one account."""
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "UPDATE reward_events SET user_id = ? WHERE user_id = ?",
                (user_id, LEGACY_USER),
            )
            old = conn.execute(
                "SELECT activity_date, events, points FROM daily_activity "
                "WHERE user_id = ?",
                (LEGACY_USER,),
            ).fetchall()
            for row in old:
                conn.execute(
                    """
                    INSERT INTO daily_activity (user_id, activity_date, events, points)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, activity_date) DO UPDATE SET
                        events = events + excluded.events,
                        points = points + excluded.points
                    """,
                    (user_id, row["activity_date"], row["events"], row["points"]),
                )
            conn.execute("DELETE FROM daily_activity WHERE user_id = ?", (LEGACY_USER,))
    finally:
<<<<<<< HEAD
        conn.close()
=======
        conn.close()
>>>>>>> 3d0b8b78da8974291b0688d9face662d07811056
