"""HeyGen API wrapper for talking-head (UGC) scenes.

Two-step flow: upload an actor photo to get a `talking_photo_id`, then submit a
video job and poll it. Unlike the fal video models, HeyGen does not take a
duration — clip length is whatever the script takes to speak, so callers must
treat the returned clip as fixed-length rather than trimming it to a target.
"""

import os
import time
import requests

API_BASE = "https://api.heygen.com"
UPLOAD_BASE = "https://upload.heygen.com"


class HeyGenError(Exception):
    pass


def _key():
    key = os.getenv("HEYGEN_API_KEY", "").strip()
    if not key:
        raise HeyGenError(
            "HEYGEN_API_KEY is not set. Get one at app.heygen.com → Settings → API, "
            "then add it to .env locally and to Railway Variables."
        )
    return key


def _headers(content_type="application/json"):
    h = {"x-api-key": _key()}
    if content_type:
        h["Content-Type"] = content_type
    return h


def _unwrap(r, what):
    """HeyGen wraps everything in {code, data, message} and returns 200 on
    application-level errors, so the body has to be inspected, not just the
    status code."""
    if r.status_code == 401:
        raise HeyGenError("HEYGEN_API_KEY is invalid (HeyGen returned 401).")
    if r.status_code >= 400:
        raise HeyGenError(f"HeyGen {what} failed ({r.status_code}): {r.text[:400]}")
    try:
        body = r.json()
    except ValueError:
        raise HeyGenError(f"HeyGen {what} returned a non-JSON response: {r.text[:200]}")

    if isinstance(body, dict):
        err = body.get("error")
        if err:
            detail = err.get("message") if isinstance(err, dict) else str(err)
            raise HeyGenError(f"HeyGen {what} error: {detail}")
        if body.get("code") not in (None, 100, 0) and body.get("data") is None:
            raise HeyGenError(f"HeyGen {what} error: {body.get('message') or body}")
        return body.get("data", body)
    return body


# ── Account ────────────────────────────────────────────────────────────────

def get_quota():
    """Remaining API credits. Read-only, costs nothing."""
    r = requests.get(f"{API_BASE}/v2/user/remaining_quota", headers=_headers(None), timeout=30)
    data = _unwrap(r, "quota check")
    # The field has been spelled both ways across API versions.
    for k in ("remaining_quota", "remaining_credits", "quota"):
        if isinstance(data, dict) and k in data:
            return data[k]
    return None


def check_account():
    try:
        quota = get_quota()
    except HeyGenError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Could not reach HeyGen: {e}"

    if quota is None:
        return True, "HeyGen connected"
    # Quota is reported in API credits; 1 credit is worth 1 second of video.
    if float(quota) <= 0:
        return False, "HeyGen has no remaining credits. Top up at app.heygen.com."
    return True, f"HeyGen connected — {quota} credits left"


# ── Voices ─────────────────────────────────────────────────────────────────

_voices_cache = None


def list_voices(limit=60):
    """English voices, best-known first. Cached for the process lifetime."""
    global _voices_cache
    if _voices_cache is not None:
        return _voices_cache

    r = requests.get(f"{API_BASE}/v2/voices", headers=_headers(None), timeout=60)
    data = _unwrap(r, "voice list")
    raw = data.get("voices", []) if isinstance(data, dict) else []

    voices = [
        {
            "id": v.get("voice_id"),
            "name": v.get("name") or v.get("display_name") or v.get("voice_id"),
            "gender": v.get("gender"),
            "language": v.get("language"),
        }
        for v in raw
        if v.get("voice_id") and str(v.get("language", "")).lower().startswith("english")
    ]
    _voices_cache = voices[:limit]
    return _voices_cache


def default_voice_id():
    voices = list_voices()
    if not voices:
        raise HeyGenError("HeyGen returned no usable English voices.")
    for v in voices:
        if (v.get("gender") or "").lower() == "female":
            return v["id"]
    return voices[0]["id"]


# ── Actor photo ────────────────────────────────────────────────────────────

def upload_talking_photo(image_bytes, content_type="image/jpeg"):
    """Upload an actor photo. Returns a talking_photo_id.

    Note: every call creates a new photo avatar group on the account, so upload
    once per actor and reuse the id rather than re-uploading per scene.
    """
    if content_type not in ("image/jpeg", "image/png"):
        content_type = "image/jpeg"
    r = requests.post(
        f"{UPLOAD_BASE}/v1/talking_photo",
        headers={"x-api-key": _key(), "Content-Type": content_type},
        data=image_bytes,
        timeout=180,
    )
    data = _unwrap(r, "photo upload")
    tp_id = data.get("talking_photo_id") if isinstance(data, dict) else None
    if not tp_id:
        raise HeyGenError(f"HeyGen upload returned no talking_photo_id: {str(data)[:300]}")
    return tp_id


# ── Video generation ───────────────────────────────────────────────────────

DIMENSIONS = {
    "9:16": {"width": 1080, "height": 1920},
    "1:1": {"width": 1080, "height": 1080},
    "16:9": {"width": 1920, "height": 1080},
}


def generate_video(talking_photo_id, script, voice_id=None, aspect_ratio="9:16",
                   on_status=None, timeout=1200, poll_every=5):
    """Render a talking-head clip and return its URL.

    HeyGen derives the length from the script, so the caller gets whatever
    duration the speech needs.
    """
    voice_id = voice_id or default_voice_id()
    payload = {
        "video_inputs": [{
            "character": {
                "type": "talking_photo",
                "talking_photo_id": talking_photo_id,
                "scale": 1.0,
            },
            "voice": {
                "type": "text",
                "input_text": script,
                "voice_id": voice_id,
            },
        }],
        "dimension": DIMENSIONS.get(aspect_ratio, DIMENSIONS["9:16"]),
        # test=true is free but stamps a watermark across the output.
        "test": False,
    }

    r = requests.post(f"{API_BASE}/v2/video/generate", headers=_headers(),
                      json=payload, timeout=120)
    data = _unwrap(r, "video generate")
    video_id = data.get("video_id") if isinstance(data, dict) else None
    if not video_id:
        raise HeyGenError(f"HeyGen returned no video_id: {str(data)[:300]}")

    if on_status:
        on_status(f"queued ({video_id[:8]})")

    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        time.sleep(poll_every)
        s = requests.get(f"{API_BASE}/v1/video_status.get",
                         headers=_headers(None), params={"video_id": video_id}, timeout=60)
        info = _unwrap(s, "status check")
        status = info.get("status") if isinstance(info, dict) else None

        if status != last:
            last = status
            if on_status:
                on_status(str(status))

        if status == "completed":
            url = info.get("video_url")
            if not url:
                raise HeyGenError("HeyGen reported completed but returned no video_url")
            return url
        if status == "failed":
            err = info.get("error") or {}
            detail = err.get("message") if isinstance(err, dict) else str(err)
            raise HeyGenError(f"HeyGen generation failed: {detail or 'no detail given'}")

    raise HeyGenError(f"HeyGen job timed out after {timeout}s")
