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
    context = kb.get_context()
    messages = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        reply, _ = agent.chat(messages, extra_context=context)
    except Exception as e:
        reply = f"Error: {str(e)}"

    kb.save_message("assistant", reply)
    return jsonify({"reply": reply})


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


@app.route("/api/knowledge", methods=["POST"])
def add_knowledge():
    if "file" in request.files:
        f = request.files["file"]
        content = f.read().decode("utf-8", errors="ignore")
        kb.add_knowledge(f.filename, content)
        return jsonify({"ok": True, "name": f.filename})
    data = request.json or {}
    name = data.get("name", "Manual")
    content = data.get("content", "")
    if not content:
        return jsonify({"error": "No content"}), 400
    kb.add_knowledge(name, content)
    return jsonify({"ok": True})


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


@app.route("/api/dev-chat", methods=["POST"])
def dev_chat():
    data = request.json
    messages = data.get("messages", [])
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    messages.append({"role": "user", "content": user_message})

    def generate():
        try:
            for event in dev_agent.chat_stream(messages):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            yield f"data: {json.dumps({'type': 'done', 'reply': f'Error: {e}', 'messages': messages})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
