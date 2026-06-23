import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
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
    conn.close()


def add_knowledge(name, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO knowledge (name, content) VALUES (?, ?)", (name, content))
    conn.commit()
    conn.close()


def list_knowledge():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name, created_at FROM knowledge ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


def delete_knowledge(knowledge_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
    conn.commit()
    conn.close()


def get_context():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT name, content FROM knowledge ORDER BY created_at DESC").fetchall()
    conn.close()
    if not rows:
        return ""
    parts = [f"### {r[0]}\n{r[1]}" for r in rows]
    return "\n\n".join(parts)


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
