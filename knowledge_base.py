import sqlite3
import os
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'General',
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    # Migration: add category column to existing databases
    try:
        conn.execute("ALTER TABLE knowledge ADD COLUMN category TEXT NOT NULL DEFAULT 'General'")
        conn.commit()
    except Exception:
        pass
    conn.close()


def add_knowledge(name, content, category="General"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO knowledge (name, content, category) VALUES (?, ?, ?)",
                 (name, content, category))
    conn.commit()
    conn.close()


def list_knowledge():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, name, category, created_at FROM knowledge ORDER BY category, created_at"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "category": r[2] or "General", "created_at": r[3]} for r in rows]


def list_categories():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT DISTINCT category FROM knowledge ORDER BY category").fetchall()
    conn.close()
    cats = [r[0] or "General" for r in rows]
    if "General" not in cats:
        cats.insert(0, "General")
    return cats


def delete_knowledge(knowledge_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
    conn.commit()
    conn.close()


def get_context():
    """Return only General category context (backward compat)."""
    return get_context_for_message("")


def get_context_for_message(message):
    """Return General context + any category that matches the message."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, name, category, content FROM knowledge ORDER BY category, created_at"
    ).fetchall()
    conn.close()

    by_cat = defaultdict(list)
    for _, name, cat, content in rows:
        by_cat[cat or "General"].append(f"### {name}\n{content}")

    msg_lower = message.lower()
    result_parts = []

    for cat, contents in by_cat.items():
        if cat == "General":
            result_parts.extend(contents)
        else:
            # Match if any meaningful word from the category name appears in the message
            words = [w for w in cat.lower().replace("-", " ").split() if len(w) > 2]
            if words and any(w in msg_lower for w in words):
                result_parts.extend(contents)

    return "\n\n".join(result_parts) if result_parts else ""


def save_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()


def get_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM chat_history ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
