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
    os.makedirs(os.path.join(_DATA_DIR, "generated_videos"), exist_ok=True)
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
    conn.execute("""CREATE TABLE IF NOT EXISTS product_pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT NOT NULL,
        title TEXT,
        price TEXT,
        shopify_product_id TEXT,
        shopify_product_url TEXT,
        admin_url TEXT,
        template_suffix TEXT,
        generated_text TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS custom_sections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        section_name TEXT NOT NULL,
        asset_key TEXT NOT NULL,
        reference_url TEXT,
        theme_id TEXT,
        liquid_code TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS video_projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        product_name TEXT,
        recipe_json TEXT NOT NULL,
        video_model TEXT,
        filename TEXT,
        scene_urls TEXT,
        clips_json TEXT,
        status TEXT NOT NULL DEFAULT 'recipe',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    for table, col, definition in [
        ("knowledge", "category", "TEXT NOT NULL DEFAULT 'General'"),
        ("knowledge", "content_type", "TEXT NOT NULL DEFAULT 'text'"),
        ("knowledge", "file_path", "TEXT"),
        # Added after video_projects shipped, so existing databases need it too.
        ("video_projects", "clips_json", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
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


def save_product_page(product_name, generated_text, title=None, price=None,
                      shopify_product_id=None, shopify_product_url=None,
                      admin_url=None, template_suffix=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO product_pages
        (product_name, generated_text, title, price, shopify_product_id,
         shopify_product_url, admin_url, template_suffix)
        VALUES (?,?,?,?,?,?,?,?)""",
        (product_name, generated_text, title, price, shopify_product_id,
         shopify_product_url, admin_url, template_suffix))
    conn.commit()
    conn.close()


def list_product_pages():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, product_name, title, price, shopify_product_id, shopify_product_url, admin_url, template_suffix, generated_text, created_at FROM product_pages ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "product_name": r[1], "title": r[2], "price": r[3],
             "shopify_product_id": r[4], "shopify_product_url": r[5],
             "admin_url": r[6], "template_suffix": r[7],
             "generated_text": r[8], "created_at": r[9]} for r in rows]


def delete_product_page(page_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM product_pages WHERE id = ?", (page_id,))
    conn.commit()
    conn.close()


def save_custom_section(section_name, asset_key, liquid_code, reference_url=None, theme_id=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO custom_sections
        (section_name, asset_key, reference_url, theme_id, liquid_code)
        VALUES (?,?,?,?,?)""",
        (section_name, asset_key, reference_url, theme_id, liquid_code))
    conn.commit()
    conn.close()


def list_custom_sections():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, section_name, asset_key, reference_url, theme_id, liquid_code, created_at "
        "FROM custom_sections ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "section_name": r[1], "asset_key": r[2], "reference_url": r[3],
             "theme_id": r[4], "liquid_code": r[5], "created_at": r[6]} for r in rows]


def get_custom_section(section_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT id, section_name, asset_key, reference_url, theme_id, liquid_code, created_at "
        "FROM custom_sections WHERE id = ?", (section_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "section_name": row[1], "asset_key": row[2], "reference_url": row[3],
            "theme_id": row[4], "liquid_code": row[5], "created_at": row[6]}


def delete_custom_section(section_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM custom_sections WHERE id = ?", (section_id,))
    conn.commit()
    conn.close()


def update_custom_section_code(section_id, liquid_code):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE custom_sections SET liquid_code = ? WHERE id = ?", (liquid_code, section_id))
    conn.commit()
    conn.close()


VIDEOS_DIR = os.path.join(_DATA_DIR, "generated_videos")


def save_video_project(title, recipe_json, product_name=None, video_model=None,
                       filename=None, scene_urls=None, status="recipe"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("""INSERT INTO video_projects
        (title, product_name, recipe_json, video_model, filename, scene_urls, status)
        VALUES (?,?,?,?,?,?,?)""",
        (title, product_name, recipe_json, video_model, filename, scene_urls, status))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_video_project(project_id, **fields):
    allowed = {"title", "recipe_json", "video_model", "filename", "scene_urls",
               "clips_json", "status"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    conn = sqlite3.connect(DB_PATH)
    clause = ", ".join(f"{k} = ?" for k in sets)
    conn.execute(f"UPDATE video_projects SET {clause} WHERE id = ?",
                 (*sets.values(), project_id))
    conn.commit()
    conn.close()


def _video_row(r):
    return {"id": r[0], "title": r[1], "product_name": r[2], "recipe_json": r[3],
            "video_model": r[4], "filename": r[5], "scene_urls": r[6],
            "clips_json": r[7], "status": r[8], "created_at": r[9]}


_VIDEO_COLS = ("id, title, product_name, recipe_json, video_model, filename, "
               "scene_urls, clips_json, status, created_at")


def list_video_projects():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        # created_at only has second precision, so it ties for videos made in
        # quick succession — id breaks the tie and matches insertion order.
        f"SELECT {_VIDEO_COLS} FROM video_projects ORDER BY created_at DESC, id DESC"
    ).fetchall()
    conn.close()
    return [_video_row(r) for r in rows]


def get_video_project(project_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        f"SELECT {_VIDEO_COLS} FROM video_projects WHERE id = ?", (project_id,)
    ).fetchone()
    conn.close()
    return _video_row(row) if row else None


def delete_video_project(project_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT filename FROM video_projects WHERE id = ?", (project_id,)).fetchone()
    if row and row[0]:
        path = os.path.join(VIDEOS_DIR, row[0])
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    conn.execute("DELETE FROM video_projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()


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
