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
    "kling-2.6-pro": {
        "id": "fal-ai/kling-video/v2.6/pro/image-to-video",
        "label": "Kling 2.6 Pro — $0.07/s ★ most realistic for the money",
        "durations": ["5", "10"],
        "usd_per_second": 0.07,
        "recommended": True,
        "note": "Newest Kling. Best bet for footage that does not read as AI: "
                "steady faces and hands, believable physics.",
    },
    "kling-2.5-pro": {
        "id": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
        "label": "Kling 2.5 Turbo Pro — $0.07/s (proven, slightly older)",
        "durations": ["5", "10"],
        "usd_per_second": 0.07,
        "note": "Same price as 2.6. Fall back here if 2.6 misreads a shot.",
    },
    "seedance-pro": {
        "id": "bytedance/seedance-2.0/image-to-video",
        "label": "Seedance 2.0 — $0.30/s (highest fidelity, 4x the price)",
        "durations": ["4", "6", "8", "10", "12"],
        "usd_per_second": 0.3034,
        "note": "Sharpest detail and best complex motion, but four times Kling Pro. "
                "Worth it for a hero shot, wasteful for testing.",
    },
    "seedance-fast": {
        "id": "bytedance/seedance-2.0/fast/image-to-video",
        "label": "Seedance 2.0 Fast — $0.24/s",
        "durations": ["4", "6", "8", "10", "12"],
        "usd_per_second": 0.2419,
        "note": "Cheaper Seedance, still 3x Kling Pro. Rarely the right pick.",
    },
    "kling-2.5-standard": {
        "id": "fal-ai/kling-video/v2.5-turbo/standard/image-to-video",
        "label": "Kling 2.5 Standard — $0.042/s (cheapest, visibly weaker)",
        "durations": ["5", "10"],
        "usd_per_second": 0.042,
        "note": "For drafting the shot list cheaply. Motion is looser and it "
                "reads as AI more often — not for a final ad.",
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
# A product swap averages more than one rung of EDIT_LADDER before it passes
# both checks, and can reach $0.23 if it goes all the way to the top.
USD_PER_EDIT = 0.08
USD_PER_TTS_LINE = 0.02

# Voices exposed by fal-ai/ai-avatar/single-text (ElevenLabs voice names).
AVATAR_VOICES = [
    "Aria", "Roger", "Sarah", "Laura", "Charlie", "George", "Callum", "River",
    "Liam", "Charlotte", "Alice", "Matilda", "Will", "Jessica", "Eric",
    "Chris", "Brian", "Daniel", "Lily", "Bill",
]

IMAGE_MODEL = "fal-ai/flux/dev"
# Image *editing* — takes real reference photos plus an instruction and keeps
# the subject intact. Text-to-image cannot reproduce a specific physical
# product no matter how precisely it is described, so every shot that has to
# show the actual product goes through here instead.
EDIT_MODEL = "fal-ai/nano-banana/edit"

# Swapping a product into an existing frame without redrawing the scene is the
# hardest step in the pipeline, and the cheap model succeeds only sometimes —
# the same prompt on the same inputs alternates between a clean swap, a no-op,
# and a wholly recomposed scene. So it is an escalating ladder, cheapest first,
# each rung verified before it is accepted.
EDIT_LADDER = [
    # Built for exactly this: it understands "the product in Figure 1 becomes
    # the one in Figure 2" as a spatial instruction across reference images.
    {"id": "fal-ai/bytedance/seedream/v4.5/edit", "label": "Seedream 4.5 Edit", "usd": 0.04},
    {"id": "fal-ai/nano-banana/edit", "label": "Nano Banana Edit", "usd": 0.04},
    # Strongest instruction follower of the three, and the only one that leaves
    # untouched regions genuinely untouched rather than resynthesised.
    {"id": "openai/gpt-image-2/edit", "label": "GPT Image 2 Edit", "usd": 0.21},
    {"id": "fal-ai/nano-banana-pro/edit", "label": "Nano Banana Pro Edit", "usd": 0.15},
]

# Video-to-audio. Generates ambience that matches what is on screen — water,
# wind, footsteps — which is most of what makes a clip read as real footage
# rather than a render. At $0.001/s it is the cheapest thing in the pipeline.
AUDIO_MODEL = "fal-ai/mmaudio-v2"
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

def _billable_duration(seconds, allowed):
    """Shortest allowed clip length that still covers the scene.

    Must round UP, not to the nearest: a 7s scene on a model offering 5s and
    10s would otherwise get a 5s clip and come back two seconds short, since
    assembly can trim a long clip but cannot invent footage for a short one.
    """
    options = sorted(allowed, key=float)
    for d in options:
        if float(d) >= float(seconds) - 0.01:
            return d
    return options[-1]


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


_SEEDREAM_SIZES = {
    "9:16": "portrait_16_9",   # ByteDance names these by the long edge
    "16:9": "landscape_16_9",
    "1:1": "square_hd",
}


def edit_image(image_urls, prompt, aspect_ratio="9:16", on_status=None,
               model_id=None, seed=None):
    """Build a shot around real reference photos. Returns image URL.

    `image_urls` are actual pictures — a frame to edit, product photos, or
    both. Which one is which is conveyed by the prompt.
    """
    if not image_urls:
        raise FalError("edit_image needs at least one reference image")
    model_id = model_id or EDIT_MODEL

    payload = {"prompt": prompt, "image_urls": image_urls[:8], "num_images": 1}
    if seed is not None:
        payload["seed"] = seed

    if "seedream" in model_id or "gpt-image" in model_id:
        # Both take a named image_size rather than an aspect ratio string.
        payload["image_size"] = _SEEDREAM_SIZES.get(aspect_ratio, "portrait_16_9")
        payload.pop("seed", None)          # gpt-image-2 rejects unknown fields
        if "gpt-image" in model_id:
            payload["quality"] = "high"
            payload["output_format"] = "jpeg"
    else:
        payload["aspect_ratio"] = aspect_ratio
        payload["output_format"] = "jpeg"

    out = run(model_id, payload, on_status=on_status, timeout=300)
    images = out.get("images") or []
    if not images:
        raise FalError(f"Edit model returned no images: {str(out)[:300]}")
    return images[0]["url"]


def generate_broll(model_key, image_url, motion_prompt, seconds, aspect_ratio="9:16", on_status=None):
    """Image-to-video. Returns video URL."""
    spec = VIDEO_MODELS.get(model_key)
    if not spec:
        raise FalError(f"Unknown video model '{model_key}'")
    duration = _billable_duration(seconds, spec["durations"])

    negative = ("blur, distort, warped hands, extra fingers, low quality, "
                "watermark, text artifacts")

    if model_key == "kling-2.6-pro":
        # 2.6 renamed the input image field. Sending `image_url` here is
        # accepted and silently ignored, which quietly downgrades the call to
        # text-to-video and throws away the starter frame.
        # generate_audio defaults to true and doubles the rate ($0.07 -> $0.14
        # per second) for a track the assembler overwrites anyway.
        payload = {
            "prompt": motion_prompt,
            "start_image_url": image_url,
            "duration": duration,
            "negative_prompt": negative,
            "generate_audio": False,
        }
    elif spec["id"].startswith("fal-ai/kling-video"):
        payload = {
            "prompt": motion_prompt,
            "image_url": image_url,
            "duration": duration,
            "negative_prompt": negative,
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
    """ElevenLabs TTS. Returns audio URL.

    Left on defaults this reads flat and synthetic. Lower stability lets v3
    actually perform the line instead of holding a neutral tone, and a little
    style pushes it further from newsreader delivery towards how someone
    talks in a UGC ad.
    """
    out = run(TTS_MODEL, {
        "text": text,
        "voice": voice if voice in AVATAR_VOICES else "Sarah",
        "stability": 0.35,
        "similarity_boost": 0.75,
        "style": 0.45,
        "speed": 1.0,
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


def generate_ambient(video_url, prompt, duration, on_status=None):
    """Generate diegetic sound matched to what happens on screen.

    Returns the URL of a video with the ambience muxed in — the caller pulls
    the audio track out of it. Costs about a cent for a short ad.
    """
    out = run(AUDIO_MODEL, {
        "video_url": video_url,
        "prompt": prompt,
        "negative_prompt": "music, soundtrack, speech, voice, narration, talking",
        "duration": float(duration),
        "num_steps": 25,
        "cfg_strength": 4.5,
    }, on_status=on_status, timeout=600)
    video = out.get("video") or {}
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        raise FalError(f"Audio model returned nothing usable: {str(out)[:300]}")
    return url


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
