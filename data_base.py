import sqlite3

DB_NAME = "recycling.db"


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

    connection.commit()
    connection.close()


def save_event(material, item_name, timestamp):
    connection = sqlite3.connect(DB_NAME)

    connection.execute(
        """
        INSERT INTO recycling_events
        (material, item_name, timestamp)
        VALUES (?, ?, ?)
        """,
        (material, item_name, timestamp)
    )

    connection.commit()
    connection.close()


def get_events():
    connection = sqlite3.connect(DB_NAME)
    connection.row_factory = sqlite3.Row

    rows = connection.execute(
        "SELECT * FROM recycling_events ORDER BY id DESC"
    ).fetchall()

    connection.close()
    return [dict(row) for row in rows]

def get_bin_totals():
    connection = sqlite3.connect(DB_NAME)

    rows = connection.execute("""
        SELECT material, COUNT(*) AS total
        FROM recycling_events
        GROUP BY material
    """).fetchall()

    connection.close()

    return dict(rows)