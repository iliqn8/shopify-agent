from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv
import os
import subprocess
import socket
import threading
import requests as _req

load_dotenv()

import knowledge_base as kb
import claude_agent as agent
import dev_agent

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "generated_images"), exist_ok=True)

kb.init_db()

# ── Start image generator in background thread ─────────────────────────────

def _start_imagegen():
    try:
        from imagegen_server import run_imagegen
        run_imagegen(port=5001)
    except Exception as e:
        print(f"[imagegen] Failed to start: {e}")

threading.Thread(target=_start_imagegen, daemon=True).start()


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("app.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    kb.save_message("user", user_message)
    history = kb.get_history(limit=20)
    context = kb.get_context_for_message(user_message)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        reply, _ = agent.chat(messages, extra_context=context)
    except Exception as e:
        reply = f"Error: {str(e)}"

    kb.save_message("assistant", reply)
    return jsonify({"reply": reply})


# ── Streaming chat with live status updates ────────────────────────────────

_chat_jobs = {}  # job_id -> {"events": [], "done": False}


@app.route("/api/chat-start", methods=["POST"])
def chat_start():
    import base64 as _b64, uuid as _uuid2
    data = request.json
    user_message = data.get("message", "").strip()

    # Support both single image (legacy) and multiple images array
    images_raw = data.get("images") or []
    if not images_raw and data.get("image_b64"):
        images_raw = [{"b64": data["image_b64"], "filename": data.get("image_filename", "image.jpg")}]

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    kb.save_message("user", user_message)
    history = kb.get_history(limit=20)
    context = kb.get_context_for_message(user_message)

    MEDIA_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                   "png": "image/png", "gif": "image/gif", "webp": "image/webp"}

    # Build messages list; inject images if provided
    messages = []
    for m in history[:-1]:
        messages.append({"role": m["role"], "content": m["content"]})

    if images_raw:
        content = []
        for img in images_raw:
            ext = img["filename"].rsplit(".", 1)[-1].lower()
            media_type = MEDIA_TYPES.get(ext, "image/jpeg")
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img["b64"]}})
        content.append({"type": "text", "text": user_message})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_message})

    # Inject training images for matching categories (max 3, before user images)
    training_imgs = kb.get_images_for_message(user_message)
    if training_imgs:
        last = messages[-1]
        if isinstance(last.get("content"), str):
            base_content = [{"type": "text", "text": last["content"]}]
        else:
            base_content = list(last.get("content", []))
        extra = []
        for img in training_imgs[:3]:
            extra.append({"type": "image", "source": {
                "type": "base64", "media_type": img["media_type"], "data": img["b64"]
            }})
        if extra:
            extra.append({"type": "text", "text": "[Reference images from training — use as visual guide]"})
            messages[-1] = {**last, "content": extra + base_content}

    # Save each attached image and pass URLs to agent context
    if images_raw:
        port = os.getenv("PORT", "5001")
        attach_urls = []
        for img in images_raw:
            ext = img["filename"].rsplit(".", 1)[-1].lower() or "jpg"
            saved_filename = f"user_attach_{_uuid2.uuid4().hex[:8]}.{ext}"
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
            with open(saved_path, "wb") as _f:
                _f.write(_b64.b64decode(img["b64"]))
            attach_urls.append(f"http://127.0.0.1:{port}/user-uploads/{saved_filename}")
        urls_str = ", ".join(attach_urls)
        context = (context or "") + f"\n\n[USER ATTACHED {len(attach_urls)} IMAGE(S) — use these URLs with upload_images_to_product: {urls_str}]"

    import uuid
    job_id = str(uuid.uuid4())
    _chat_jobs[job_id] = {"events": [], "done": False}

    def run():
        try:
            for event in agent.chat_stream(messages, extra_context=context):
                _chat_jobs[job_id]["events"].append(event)
                if event.get("type") == "done":
                    reply = event.get("reply", "")
                    kb.save_message("assistant", reply)
                    _chat_jobs[job_id]["done"] = True
        except Exception as e:
            err = f"Error: {e}"
            kb.save_message("assistant", err)
            _chat_jobs[job_id]["events"].append({"type": "done", "reply": err, "messages": []})
            _chat_jobs[job_id]["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/chat-poll/<job_id>")
def chat_poll(job_id):
    job = _chat_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    since = request.args.get("since", 0, type=int)
    events = job["events"][since:]
    done = job["done"]
    # Clean up finished jobs after a delay
    if done and since + len(events) >= len(job["events"]):
        threading.Timer(60, lambda: _chat_jobs.pop(job_id, None)).start()
    return jsonify({"events": events, "total": len(job["events"]), "done": done})


@app.route("/api/history", methods=["GET"])
def history():
    return jsonify(kb.get_history(limit=100))


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    kb.clear_history()
    return jsonify({"ok": True})


@app.route("/api/knowledge", methods=["GET"])
def list_knowledge():
    return jsonify(kb.list_knowledge())


_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}


def _read_uploaded_file(f):
    filename = f.filename.lower()
    if filename.endswith(".docx"):
        try:
            import docx, io
            doc = docx.Document(io.BytesIO(f.read()))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            return f"[Could not parse docx: {e}]"
    return f.read().decode("utf-8", errors="ignore")


@app.route("/api/knowledge", methods=["POST"])
def add_knowledge():
    import uuid as _uuid3
    category = request.args.get("category", "General")
    if "file" in request.files:
        f = request.files["file"]
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext in _IMAGE_EXTS:
            # Save image to persistent storage
            os.makedirs(kb.IMAGES_DIR, exist_ok=True)
            filename = f"training_{_uuid3.uuid4().hex[:8]}.{ext}"
            save_path = os.path.join(kb.IMAGES_DIR, filename)
            f.save(save_path)
            kb.add_image(f.filename, save_path, category)
            return jsonify({"ok": True, "name": f.filename, "type": "image"})
        content = _read_uploaded_file(f)
        kb.add_knowledge(f.filename, content, category)
        return jsonify({"ok": True, "name": f.filename})
    data = request.json or {}
    name = data.get("name", "Manual")
    content = data.get("content", "")
    category = data.get("category", category)
    if not content:
        return jsonify({"error": "No content"}), 400
    kb.add_knowledge(name, content, category)
    return jsonify({"ok": True})


@app.route("/api/categories", methods=["GET"])
def get_categories():
    return jsonify(kb.list_categories())


@app.route("/api/knowledge/<int:kid>", methods=["DELETE"])
def delete_knowledge(kid):
    kb.delete_knowledge(kid)
    return jsonify({"ok": True})


import time as _time
_START_TIME = str(int(_time.time()))

@app.route("/api/version")
def version():
    v = os.getenv("RAILWAY_GIT_COMMIT_SHA", _START_TIME)
    return jsonify({"version": v})


@app.route("/api/network-info")
def network_info():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"
    port = os.getenv("PORT", "5001")
    return jsonify({"local_ip": local_ip, "port": port, "url": f"http://{local_ip}:{port}"})


@app.route("/api/start-imagegen", methods=["POST"])
def start_imagegen():
    return jsonify({"status": "running"})


_dev_jobs = {}  # job_id -> {"events": [], "done": False}


@app.route("/api/dev-chat", methods=["POST"])
def dev_chat():
    data = request.json
    messages = data.get("messages", [])
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    image_b64 = data.get("image_b64")
    image_filename = data.get("image_filename", "image.jpg")
    if image_b64:
        import re as _re
        ext = image_filename.rsplit(".", 1)[-1].lower() if "." in image_filename else "jpeg"
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
        messages.append({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}},
            {"type": "text", "text": user_message},
        ]})
    else:
        messages.append({"role": "user", "content": user_message})

    import uuid, threading as _th
    job_id = str(uuid.uuid4())
    _dev_jobs[job_id] = {"events": [{"type": "status", "text": "🤔 Thinking..."}], "done": False}

    def run():
        try:
            for event in dev_agent.chat_stream(messages):
                if event.get("type") != "ping":
                    _dev_jobs[job_id]["events"].append(event)
                if event.get("type") == "done":
                    _dev_jobs[job_id]["done"] = True
        except Exception as e:
            _dev_jobs[job_id]["events"].append({"type": "done", "reply": f"Error: {e}", "messages": messages})
            _dev_jobs[job_id]["done"] = True

    _th.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/dev-poll/<job_id>")
def dev_poll(job_id):
    job = _dev_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    since = request.args.get("since", 0, type=int)
    events = job["events"][since:]
    if job["done"] and not _dev_jobs.get(job_id + "_keep"):
        # Clean up after done is fully consumed
        if since + len(events) >= len(job["events"]):
            threading.Timer(30, lambda: _dev_jobs.pop(job_id, None)).start()
    return jsonify({"events": events, "total": len(job["events"]), "done": job["done"]})


_build_jobs = {}


@app.route("/api/product-build-start", methods=["POST"])
def product_build_start():
    import uuid as _uuid4
    data = request.json or {}
    product_name = data.get("product_name", "").strip()
    competitor_url = data.get("competitor_url", "").strip()
    product_cost = data.get("product_cost", "0")
    shipping_cost = data.get("shipping_cost", "0")
    images = data.get("images", [])

    if not product_name:
        return jsonify({"error": "Product name required"}), 400

    job_id = str(_uuid4.uuid4())
    _build_jobs[job_id] = {"events": [], "done": False}

    def run():
        try:
            import product_builder
            for event in product_builder.build_stream(
                product_name, competitor_url, product_cost, shipping_cost, images
            ):
                _build_jobs[job_id]["events"].append(event)
                if event.get("type") == "done":
                    _build_jobs[job_id]["done"] = True
        except Exception as e:
            _build_jobs[job_id]["events"].append({"type": "done", "reply": f"Error: {e}"})
            _build_jobs[job_id]["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/product-publish", methods=["POST"])
def product_publish():
    import product_publisher as pub
    data = request.json or {}
    product_name = data.get("product_name", "").strip()
    generated_text = data.get("generated_text", "").strip()
    if not product_name or not generated_text:
        return jsonify({"error": "Missing fields"}), 400
    try:
        result = pub.publish(product_name, generated_text)
        kb.save_product_page(
            product_name=product_name,
            generated_text=generated_text,
            title=result.get("title"),
            price=result.get("price"),
            shopify_product_id=result.get("product_id"),
            shopify_product_url=result.get("product_url"),
            admin_url=result.get("admin_url"),
            template_suffix=result.get("template_suffix"),
        )
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/product-pages", methods=["GET"])
def list_product_pages():
    return jsonify(kb.list_product_pages())


@app.route("/api/product-pages/<int:pid>", methods=["DELETE"])
def delete_product_page(pid):
    kb.delete_product_page(pid)
    return jsonify({"ok": True})


@app.route("/api/backup-db")
def backup_db():
    from flask import send_file
    db_path = kb.DB_PATH
    if not os.path.exists(db_path):
        return jsonify({"error": "DB not found"}), 404
    return send_file(db_path, as_attachment=True, download_name="knowledge_backup.db")


@app.route("/api/restore-db", methods=["POST"])
def restore_db():
    import shutil
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    backup_path = kb.DB_PATH + ".bak"
    shutil.copy2(kb.DB_PATH, backup_path)
    f.save(kb.DB_PATH)
    kb.init_db()
    return jsonify({"ok": True})


@app.route("/api/product-build-poll/<job_id>")
def product_build_poll(job_id):
    job = _build_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    since = request.args.get("since", 0, type=int)
    events = job["events"][since:]
    done = job["done"]
    if done and since + len(events) >= len(job["events"]):
        threading.Timer(120, lambda: _build_jobs.pop(job_id, None)).start()
    return jsonify({"events": events, "total": len(job["events"]), "done": done})


_section_jobs = {}


@app.route("/api/section-build-start", methods=["POST"])
def section_build_start():
    import uuid as _uuid5
    data = request.json or {}
    reference_url = data.get("reference_url", "").strip()
    section_name = data.get("section_name", "").strip() or None
    image = data.get("image")

    if not reference_url:
        return jsonify({"error": "Reference URL required"}), 400
    if not image or not image.get("b64"):
        return jsonify({"error": "Screenshot required"}), 400

    job_id = str(_uuid5.uuid4())
    _section_jobs[job_id] = {"events": [], "done": False}

    def run():
        try:
            import section_builder
            for event in section_builder.build_stream(reference_url, image, section_name):
                _section_jobs[job_id]["events"].append(event)
                if event.get("type") == "done":
                    _section_jobs[job_id]["done"] = True
        except Exception as e:
            _section_jobs[job_id]["events"].append({"type": "done", "reply": f"Error: {e}"})
            _section_jobs[job_id]["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/section-build-poll/<job_id>")
def section_build_poll(job_id):
    job = _section_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    since = request.args.get("since", 0, type=int)
    events = job["events"][since:]
    done = job["done"]
    if done and since + len(events) >= len(job["events"]):
        threading.Timer(120, lambda: _section_jobs.pop(job_id, None)).start()
    return jsonify({"events": events, "total": len(job["events"]), "done": done})


@app.route("/api/section-publish", methods=["POST"])
def section_publish():
    import section_builder
    import shopify_client as sc
    data = request.json or {}
    section_name = data.get("section_name", "").strip() or "Custom Section"
    liquid_code = data.get("liquid_code", "").strip()
    reference_url = data.get("reference_url", "").strip()

    if not liquid_code:
        return jsonify({"error": "Missing liquid_code"}), 400

    try:
        theme = sc.get_active_theme()
        if not theme:
            return jsonify({"error": "No active theme found"}), 500
        theme_id = theme["id"]
        asset_key = section_builder.unique_asset_key(section_name)
        sc.update_theme_file(theme_id, asset_key, liquid_code)
        kb.save_custom_section(
            section_name=section_name,
            asset_key=asset_key,
            liquid_code=liquid_code,
            reference_url=reference_url,
            theme_id=str(theme_id),
        )
        return jsonify({
            "ok": True,
            "asset_key": asset_key,
            "theme_id": theme_id,
            "editor_url": f"https://{sc.SHOP}/admin/themes/{theme_id}/editor",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/custom-sections", methods=["GET"])
def list_custom_sections():
    return jsonify(kb.list_custom_sections())


@app.route("/api/custom-sections/<int:sid>", methods=["DELETE"])
def delete_custom_section(sid):
    import shopify_client as sc
    row = kb.get_custom_section(sid)
    if row and row.get("theme_id") and row.get("asset_key"):
        try:
            sc.delete_theme_file(row["theme_id"], row["asset_key"])
        except Exception as e:
            print(f"[section_builder] Failed to delete theme file: {e}")
    kb.delete_custom_section(sid)
    return jsonify({"ok": True})


@app.route("/api/set-local-agent", methods=["POST"])
def set_local_agent():
    url = request.json.get("url", "")
    os.environ["LOCAL_AGENT_URL"] = url
    print(f"[server] Local agent URL set: {url}")
    return jsonify({"ok": True})


# ── Image Generator Proxy ──────────────────────────────────────────────────

IMG_GEN = "http://localhost:5001"


def _proxy(path, method, stream=False):
    url = f"{IMG_GEN}/{path}"
    if method == "POST":
        if request.content_type and "multipart" in request.content_type:
            r = _req.post(url, data=request.form, files={
                k: (v.filename, v.stream, v.content_type)
                for k, v in request.files.items()
            }, timeout=30)
        elif request.is_json:
            r = _req.post(url, json=request.get_json(), stream=True, timeout=300)
        else:
            r = _req.post(url, data=request.get_data(), timeout=30)
    else:
        r = _req.get(url, params=request.args, stream=stream, timeout=60)
    return r


@app.route("/imagegen")
@app.route("/imagegen/")
def imagegen_root():
    r = _proxy("", "GET")
    return Response(r.content, content_type=r.headers.get("Content-Type", "text/html"))


@app.route("/imagegen/<path:path>", methods=["GET", "POST"])
def imagegen_proxy(path):
    stream = request.method == "GET" and "generate" in path
    r = _proxy(path, request.method, stream=stream)
    ct = r.headers.get("Content-Type", "application/octet-stream")
    if "event-stream" in ct:
        return Response(stream_with_context(r.iter_content(chunk_size=None)),
                        content_type=ct,
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return Response(r.content, content_type=ct, status=r.status_code)


@app.route("/user-uploads/<filename>")
def serve_user_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/uploads/<path:filename>")
def proxy_uploads(filename):
    r = _req.get(f"{IMG_GEN}/uploads/{filename}", timeout=10)
    return Response(r.content, content_type=r.headers.get("Content-Type", "image/jpeg"))


@app.route("/generated/<path:filename>")
def proxy_generated(filename):
    r = _req.get(f"{IMG_GEN}/generated/{filename}", timeout=10)
    return Response(r.content, content_type=r.headers.get("Content-Type", "image/png"))


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print(f"Shopify AI Agent running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
