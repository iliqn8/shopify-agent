"""
Local Computer Control Agent
Runs on this PC and exposes computer control to the Railway-hosted agent via ngrok.
"""
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import subprocess
import requests
import time
import threading

load_dotenv()

app = Flask(__name__)
RAILWAY_URL = os.getenv("RAILWAY_URL", "").rstrip("/")
LOCAL_PORT = 5002


@app.route("/run-tool", methods=["POST"])
def run_tool():
    data = request.json
    tool = data.get("tool")
    inputs = data.get("inputs", {})

    try:
        if tool == "run_command":
            result = subprocess.run(
                ["powershell", "-Command", inputs["command"]],
                capture_output=True, text=True, timeout=30
            )
            return jsonify({"stdout": result.stdout[:3000], "stderr": result.stderr[:1000], "returncode": result.returncode})

        elif tool == "open_application":
            subprocess.Popen(["start", inputs["target"]], shell=True)
            return jsonify({"opened": inputs["target"]})

        elif tool == "list_files":
            result = subprocess.run(
                ["powershell", "-Command", f"Get-ChildItem '{inputs['path']}' | Select-Object Name, Length, LastWriteTime | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            return jsonify({"files": result.stdout[:3000]})

        elif tool == "read_file":
            with open(inputs["path"], "r", encoding="utf-8", errors="ignore") as f:
                return jsonify({"content": f.read(5000)})

        elif tool == "write_file":
            with open(inputs["path"], "w", encoding="utf-8") as f:
                f.write(inputs["content"])
            return jsonify({"written": inputs["path"]})

        elif tool == "get_running_processes":
            result = subprocess.run(
                ["powershell", "-Command", "Get-Process | Sort-Object CPU -Descending | Select-Object -First 30 Name, CPU, WorkingSet | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            return jsonify({"processes": result.stdout[:3000]})

        else:
            return jsonify({"error": f"Unknown tool: {tool}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


def get_ngrok_url():
    """Get public URL from ngrok API."""
    for _ in range(10):
        try:
            r = requests.get("http://localhost:4040/api/tunnels", timeout=2)
            tunnels = r.json().get("tunnels", [])
            for t in tunnels:
                if t.get("proto") == "https":
                    return t["public_url"]
        except Exception:
            pass
        time.sleep(1)
    return None


def announce_to_railway(ngrok_url):
    """Send our ngrok URL to the Railway agent."""
    if not RAILWAY_URL:
        print("[local] RAILWAY_URL not set — skipping announcement")
        return
    try:
        requests.post(
            f"{RAILWAY_URL}/api/set-local-agent",
            json={"url": ngrok_url},
            timeout=5
        )
        print(f"[local] Announced to Railway: {ngrok_url}")
    except Exception as e:
        print(f"[local] Could not announce to Railway: {e}")


def start_ngrok_and_announce():
    time.sleep(2)
    # Start ngrok
    subprocess.Popen(
        ["ngrok", "http", str(LOCAL_PORT), "--log=stdout"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(4)
    url = get_ngrok_url()
    if url:
        print(f"\n✅ Local agent public URL: {url}\n")
        announce_to_railway(url)
    else:
        print("[local] Could not get ngrok URL")


if __name__ == "__main__":
    print(f"Local Control Agent starting on port {LOCAL_PORT}...")
    threading.Thread(target=start_ngrok_and_announce, daemon=True).start()
    app.run(host="0.0.0.0", port=LOCAL_PORT, debug=False)
