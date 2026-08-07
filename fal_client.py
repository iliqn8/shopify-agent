"""Thin wrapper around the fal.ai queue API.

Everything goes through https://queue.fal.run — submit returns a request_id plus
absolute status/response URLs, which we poll rather than reconstructing paths
ourselves (nested model ids like `fal-ai/kling-video/v2.5-turbo/pro/image-to-video`
do NOT map 1:1 onto the status path, so the returned URLs are the only safe route).
"""

import os
import base64
import time
import mimetypes
import requests

QUEUE_BASE = "https://queue.fal.run"
STORAGE_INITIATE = "https://rest.alpha.fal.ai/storage/upload/initiate"


class FalError(Exception):
    pass


def _key():
    key = os.getenv("FAL_KEY", "").strip()
    if not key:
        raise FalError("FAL_KEY is not set. Add it to .env locally and to Railway Variables.")
    return key


def _headers():
    return {"Authorization": f"Key {_key()}", "Content-Type": "application/json"}


# ── Model registry ─────────────────────────────────────────────────────────
# Verified endpoint ids on fal.ai. Keys are what the UI sends us.

# `usd_per_second` is fal's published list price at the resolution we request.
# Keep these honest — the UI turns them into a pre-flight cost estimate, and the
# spread between the cheapest and dearest option here is nearly 6x.
VIDEO_MODELS = {
    "kling-2.5-standard": {
        "id": "fal-ai/kling-video/v2.5-turbo/standard/image-to-video",
        "label": "Kling 2.5 Turbo Standard — $0.042/s (cheapest)",
        "durations": ["5", "10"],
        "usd_per_second": 0.042,
    },
    "kling-2.5-pro": {
        "id": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
        "label": "Kling 2.5 Turbo Pro — $0.07/s (best motion)",
        "durations": ["5", "10"],
        "usd_per_second": 0.07,
    },
    "kling-2.6-pro": {
        "id": "fal-ai/kling-video/v2.6/pro/image-to-video",
        "label": "Kling 2.6 Pro — $0.07/s (newest)",
        "durations": ["5", "10"],
        "usd_per_second": 0.07,
    },
    "seedance-fast": {
        "id": "bytedance/seedance-2.0/fast/image-to-video",
        "label": "Seedance 2.0 Fast — $0.24/s",
        "durations": ["4", "6", "8", "10", "12"],
        "usd_per_second": 0.2419,
    },
    "seedance-pro": {
        "id": "bytedance/seedance-2.0/image-to-video",
        "label": "Seedance 2.0 — $0.30/s (720p, highest fidelity)",
        "durations": ["4", "6", "8", "10", "12"],
        "usd_per_second": 0.3034,
    },
}

AVATAR_MODELS = {
    "ai-avatar": {
        "id": "fal-ai/ai-avatar/single-text",
        "label": "AI Avatar — photo + script, built-in voice + lipsync",
        "usd_per_second": {"480p": 0.20, "720p": 0.40},
    },
    "infinitalk": {
        "id": "fal-ai/infinitalk/single-text",
        "label": "InfiniTalk — longer takes, natural expressions",
        "usd_per_second": {"480p": 0.20, "720p": 0.40},
    },
}

# Flat-ish costs for the supporting calls, used by the estimator.
USD_PER_IMAGE = 0.025
USD_PER_TTS_LINE = 0.02

# Voices exposed by fal-ai/ai-avatar/single-text (ElevenLabs voice names).
AVATAR_VOICES = [
    "Aria", "Roger", "Sarah", "Laura", "Charlie", "George", "Callum", "River",
    "Liam", "Charlotte", "Alice", "Matilda", "Will", "Jessica", "Eric",
    "Chris", "Brian", "Daniel", "Lily", "Bill",
]

IMAGE_MODEL = "fal-ai/flux/dev"
TTS_MODEL = "fal-ai/elevenlabs/tts/eleven-v3"


# ── Queue plumbing ─────────────────────────────────────────────────────────

def submit(model_id, payload):
    """Enqueue a job. Returns (request_id, status_url, response_url)."""
    r = requests.post(f"{QUEUE_BASE}/{model_id}", headers=_headers(), json=payload, timeout=60)
    if r.status_code == 403 and "balance" in r.text.lower():
        raise FalError("fal.ai account has no balance. Top up at fal.ai/dashboard/billing.")
    if r.status_code >= 400:
        raise FalError(f"fal.ai {model_id} rejected the request ({r.status_code}): {r.text[:400]}")
    data = r.json()
    return data.get("request_id"), data.get("status_url"), data.get("response_url")


def run(model_id, payload, on_status=None, timeout=1200, poll_every=3):
    """Submit and block until the job completes. Returns the result dict.

    `on_status(text)` is called on every meaningful state change so callers can
    stream progress into the UI.
    """
    request_id, status_url, response_url = submit(model_id, payload)
    if on_status:
        on_status(f"queued ({request_id[:8]})")

    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        time.sleep(poll_every)
        s = requests.get(status_url, headers={"Authorization": f"Key {_key()}"}, timeout=60)
        if s.status_code >= 400:
            raise FalError(f"fal.ai status check failed ({s.status_code}): {s.text[:300]}")
        info = s.json()
        state = info.get("status")
        if state != last:
            last = state
            if on_status:
                pos = info.get("queue_position")
                on_status(f"{state.lower().replace('_', ' ')}" + (f" (position {pos})" if pos else ""))
        if state == "COMPLETED":
            break
        if state in ("FAILED", "ERROR", "CANCELLED"):
            raise FalError(f"fal.ai job {state}: {str(info)[:400]}")
    else:
        raise FalError(f"fal.ai job timed out after {timeout}s")

    out = requests.get(response_url, headers={"Authorization": f"Key {_key()}"}, timeout=120)
    if out.status_code >= 400:
        raise FalError(f"fal.ai result fetch failed ({out.status_code}): {out.text[:300]}")
    return out.json()


# ── File input helpers ─────────────────────────────────────────────────────

def upload_bytes(data, filename="upload.jpg", content_type=None):
    """Upload raw bytes to fal storage and return a public URL.

    Falls back to a data: URI if the storage API is unavailable — fal model
    endpoints accept both for *_url fields.
    """
    content_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    try:
        init = requests.post(
            STORAGE_INITIATE,
            headers=_headers(),
            json={"content_type": content_type, "file_name": filename},
            timeout=60,
        )
        if init.status_code < 400:
            info = init.json()
            put = requests.put(
                info["upload_url"],
                data=data,
                headers={"Content-Type": content_type},
                timeout=300,
            )
            if put.status_code < 400:
                return info["file_url"]
    except Exception:
        pass
    return to_data_uri(data, content_type)


def to_data_uri(data, content_type="image/jpeg"):
    return f"data:{content_type};base64,{base64.b64encode(data).decode()}"


def download(url, timeout=600):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


# ── Payload builders (schemas differ per model family) ─────────────────────

def _nearest_duration(seconds, allowed):
    return min(allowed, key=lambda d: abs(float(d) - float(seconds)))


def generate_image(prompt, aspect_ratio="9:16", on_status=None):
    """Text-to-image starter frame. Returns image URL."""
    size = {
        "9:16": "portrait_16_9",
        "16:9": "landscape_16_9",
        "1:1": "square_hd",
    }.get(aspect_ratio, "portrait_16_9")
    out = run(IMAGE_MODEL, {
        "prompt": prompt,
        "image_size": size,
        "num_images": 1,
        "enable_safety_checker": True,
    }, on_status=on_status, timeout=300)
    images = out.get("images") or []
    if not images:
        raise FalError("Image model returned no images")
    return images[0]["url"]


def generate_broll(model_key, image_url, motion_prompt, seconds, aspect_ratio="9:16", on_status=None):
    """Image-to-video. Returns video URL."""
    spec = VIDEO_MODELS.get(model_key)
    if not spec:
        raise FalError(f"Unknown video model '{model_key}'")
    duration = _nearest_duration(seconds, spec["durations"])

    if spec["id"].startswith("fal-ai/kling-video"):
        payload = {
            "prompt": motion_prompt,
            "image_url": image_url,
            "duration": duration,
            "negative_prompt": "blur, distort, warped hands, extra fingers, low quality, watermark, text artifacts",
            "cfg_scale": 0.5,
        }
    else:  # seedance
        payload = {
            "prompt": motion_prompt,
            "image_url": image_url,
            "duration": duration,
            "resolution": "720p",
        }

    out = run(spec["id"], payload, on_status=on_status)
    video = out.get("video") or {}
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        raise FalError(f"Video model returned no video: {str(out)[:300]}")
    return url


def generate_avatar(model_key, image_url, script, voice="Sarah", seconds=6,
                    scene_prompt=None, resolution="480p", on_status=None):
    """Talking-head UGC clip. Returns video URL."""
    spec = AVATAR_MODELS.get(model_key)
    if not spec:
        raise FalError(f"Unknown avatar model '{model_key}'")

    # ai-avatar works in frames at 25fps; clamp to the documented 41-721 range.
    num_frames = max(41, min(721, int(round(float(seconds) * 25))))
    payload = {
        "image_url": image_url,
        "text_input": script,
        "voice": voice if voice in AVATAR_VOICES else "Sarah",
        "prompt": scene_prompt or "A person speaking directly to the camera, natural expressions, steady framing",
        "num_frames": num_frames,
        "resolution": resolution,
        "acceleration": "regular",
    }
    out = run(spec["id"], payload, on_status=on_status)
    video = out.get("video") or {}
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        raise FalError(f"Avatar model returned no video: {str(out)[:300]}")
    return url


def generate_voiceover(text, voice="Sarah", on_status=None):
    """ElevenLabs TTS. Returns audio URL."""
    out = run(TTS_MODEL, {
        "text": text,
        "voice": voice if voice in AVATAR_VOICES else "Sarah",
    }, on_status=on_status, timeout=300)
    audio = out.get("audio") or {}
    url = audio.get("url") if isinstance(audio, dict) else None
    if not url:
        raise FalError(f"TTS returned no audio: {str(out)[:300]}")
    return url


BALANCE_URL = "https://rest.alpha.fal.ai/billing/user_balance"


def get_balance():
    """Remaining USD credit. Read-only — costs nothing.

    Do NOT probe auth by POSTing to queue.fal.run: that endpoint does not
    validate payloads up front, it enqueues whatever it is given (even an empty
    body) and bills for the run.
    """
    r = requests.get(BALANCE_URL, headers={"Authorization": f"Key {_key()}"}, timeout=30)
    if r.status_code == 401:
        raise FalError("FAL_KEY is invalid (fal.ai returned 401).")
    if r.status_code >= 400:
        raise FalError(f"fal.ai balance check failed ({r.status_code}): {r.text[:200]}")
    return float(r.text.strip())


def check_account():
    """Returns (ok, human-readable message)."""
    try:
        balance = get_balance()
    except FalError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Could not reach fal.ai: {e}"

    if balance <= 0:
        return False, "fal.ai balance is $0. Top up at fal.ai/dashboard/billing."
    if balance < 1:
        return True, f"fal.ai balance is low: ${balance:.2f}"
    return True, f"fal.ai connected — ${balance:.2f} balance"
