import sqlite3
import threading
from datetime import datetime

DB_NAME = "recycling.db"

# Rows saved before login existed are kept under this id until claimed.
LEGACY_USER = "legacy"

# The signed-in user is stored per THREAD. Streamlit runs each visitor's
# session in its own thread, so two people using the app at the same time
# can never see each other's records (a plain global variable would mix them).
_current = threading.local()


def set_current_user(user_id):
    """Call once at the top of every script run, right after login."""
    _current.user_id = user_id


def _uid():
    user_id = getattr(_current, "user_id", None)
    if not user_id:
        # Fail closed: never fall back to showing everyone's data.
        raise RuntimeError("No signed-in user. Call set_current_user() first.")
    return user_id


def create_data_base():
    connection = sqlite3.connect(DB_NAME)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS recycling_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material TEXT NOT NULL,
            item_name TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    # Migration: add user_id to an existing database (safe to run every start)
    columns = [row[1] for row in connection.execute("PRAGMA table_info(recycling_events)")]
    if "user_id" not in columns:
        connection.execute(
            f"ALTER TABLE recycling_events "
            f"ADD COLUMN user_id TEXT NOT NULL DEFAULT '{LEGACY_USER}'"
        )

    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_user ON recycling_events(user_id)"
    )

    # Human-AI corrections table: tracks when user corrects AI suggestions
    connection.execute("""
        CREATE TABLE IF NOT EXISTS ai_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            item_name TEXT,
            suggested_label TEXT NOT NULL,
            chosen_label TEXT NOT NULL,
            suggested_bin TEXT NOT NULL,
            chosen_bin TEXT NOT NULL,
            was_corrected INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_corr_user ON ai_corrections(user_id)"
    )

    connection.commit()
    connection.close()


def save_event(material, item_name, timestamp):
    connection = sqlite3.connect(DB_NAME)

    connection.execute(
        """
        INSERT INTO recycling_events
        (material, item_name, timestamp, user_id)
        VALUES (?, ?, ?, ?)
        """,
        (material, item_name, timestamp, _uid())
    )

    connection.commit()
    connection.close()


def get_events():
    connection = sqlite3.connect(DB_NAME)
    connection.row_factory = sqlite3.Row

    rows = connection.execute(
        "SELECT * FROM recycling_events WHERE user_id = ? ORDER BY id DESC",
        (_uid(),)
    ).fetchall()

    connection.close()
    return [dict(row) for row in rows]


def get_bin_totals():
    connection = sqlite3.connect(DB_NAME)

    rows = connection.execute("""
        SELECT material, COUNT(*) AS total
        FROM recycling_events
        WHERE user_id = ?
        GROUP BY material
    """, (_uid(),)).fetchall()

    connection.close()

    return dict(rows)


def clear_events():
    """Delete ONLY the signed-in user's recycling events."""
    connection = sqlite3.connect(DB_NAME)
    connection.execute("DELETE FROM recycling_events WHERE user_id = ?", (_uid(),))
    connection.commit()
    connection.close()


# ---------- AI prediction corrections ----------
def log_correction(item_name, suggested_label, chosen_label, suggested_bin, chosen_bin):
    """Log whether the user accepted or corrected the AI recommendation."""
    connection = sqlite3.connect(DB_NAME)
    was_corrected = 1 if (
        str(suggested_label).strip().lower() != str(chosen_label).strip().lower()
        or str(suggested_bin).strip().lower() != str(chosen_bin).strip().lower()
    ) else 0
    now = datetime.now().isoformat(timespec="seconds")
    connection.execute(
        """
        INSERT INTO ai_corrections
        (user_id, item_name, suggested_label, chosen_label, suggested_bin, chosen_bin, was_corrected, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (_uid(), item_name, suggested_label, chosen_label, suggested_bin, chosen_bin, was_corrected, now)
    )
    connection.commit()
    connection.close()


def get_correction_stats():
    """Return counts and percentage of human corrections vs AI suggestions."""
    connection = sqlite3.connect(DB_NAME)
    row = connection.execute(
        """
        SELECT COUNT(*) AS total, COALESCE(SUM(was_corrected), 0) AS corrections
        FROM ai_corrections
        WHERE user_id = ?
        """,
        (_uid(),)
    ).fetchone()
    connection.close()
    total = row[0] if row else 0
    corrections = row[1] if row else 0
    rate = (corrections / total * 100.0) if total > 0 else 0.0
    return {
        "total": total,
        "corrections": corrections,
        "accepted": total - corrections,
        "correction_rate": rate,
    }


# ---------- one-time adoption of pre-login records ----------
def count_legacy_events():
    connection = sqlite3.connect(DB_NAME)
    total = connection.execute(
        "SELECT COUNT(*) FROM recycling_events WHERE user_id = ?", (LEGACY_USER,)
    ).fetchone()[0]
    connection.close()
    return total


def claim_legacy_events(user_id):
    """Assign all pre-login records to one account."""
    connection = sqlite3.connect(DB_NAME)
    connection.execute(
        "UPDATE recycling_events SET user_id = ? WHERE user_id = ?",
        (user_id, LEGACY_USER)
    )
    connection.commit()
    connection.close()

