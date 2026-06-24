import sqlite3
import os
import base64
from collections import defaultdict

# Use persistent volume on Railway (/data), fall back to local for dev
_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(__file__)
DB_PATH = os.path.join(_DATA_DIR, "knowledge.db")
IMAGES_DIR = os.path.join(_DATA_DIR, "training_images")

MEDIA_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
               "gif": "image/gif", "webp": "image/webp"}


def init_db():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'General',
        content TEXT NOT NULL DEFAULT '',
        content_type TEXT NOT NULL DEFAULT 'text',
        file_path TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    for col, definition in [
        ("category", "TEXT NOT NULL DEFAULT 'General'"),
        ("content_type", "TEXT NOT NULL DEFAULT 'text'"),
        ("file_path", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE knowledge ADD COLUMN {col} {definition}")
            conn.commit()
        except Exception:
            pass
    conn.close()


def add_knowledge(name, content, category="General"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO knowledge (name, content, category, content_type) VALUES (?, ?, ?, 'text')",
                 (name, content, category))
    conn.commit()
    conn.close()


def add_image(name, file_path, category="General"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO knowledge (name, content, category, content_type, file_path) VALUES (?, '', ?, 'image', ?)",
        (name, category, file_path)
    )
    conn.commit()
    conn.close()


def list_knowledge():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, name, category, content_type, created_at FROM knowledge ORDER BY category, created_at"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "category": r[2] or "General",
             "content_type": r[3] or "text", "created_at": r[4]} for r in rows]


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
    # Also delete image file if present
    row = conn.execute("SELECT file_path FROM knowledge WHERE id = ?", (knowledge_id,)).fetchone()
    if row and row[0] and os.path.exists(row[0]):
        try:
            os.remove(row[0])
        except Exception:
            pass
    conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
    conn.commit()
    conn.close()


def get_context():
    return get_context_for_message("")


def get_context_for_message(message):
    """Return text context for General + matching categories."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT name, category, content, content_type FROM knowledge ORDER BY category, created_at"
    ).fetchall()
    conn.close()

    by_cat = defaultdict(list)
    for name, cat, content, ctype in rows:
        if (ctype or "text") == "text" and content:
            by_cat[cat or "General"].append(f"### {name}\n{content}")

    msg_lower = message.lower()
    result_parts = []
    for cat, contents in by_cat.items():
        if cat == "General":
            result_parts.extend(contents)
        else:
            words = [w for w in cat.lower().replace("-", " ").split() if len(w) > 2]
            if words and any(w in msg_lower for w in words):
                result_parts.extend(contents)

    return "\n\n".join(result_parts) if result_parts else ""


def get_images_for_message(message):
    """Return list of {b64, media_type, name} for images in matching categories."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT name, category, file_path FROM knowledge WHERE content_type = 'image' ORDER BY category, created_at"
    ).fetchall()
    conn.close()

    msg_lower = message.lower()
    result = []
    for name, cat, file_path in rows:
        cat = cat or "General"
        if cat == "General":
            match = True
        else:
            words = [w for w in cat.lower().replace("-", " ").split() if len(w) > 2]
            match = bool(words and any(w in msg_lower for w in words))
        if match and file_path and os.path.exists(file_path):
            ext = file_path.rsplit(".", 1)[-1].lower()
            media_type = MEDIA_TYPES.get(ext, "image/jpeg")
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            result.append({"b64": b64, "media_type": media_type, "name": name})
    return result


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
