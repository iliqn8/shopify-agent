"""Motion Studio — a person from a photo performs the moves of a reference video.

Kling 3.0 Motion Control on fal. The photo decides who is on screen, what they
wear and where they stand; the video decides how they move. An optional face
photo is bound as `@Element1`, which keeps a real person's identity steadier
than the full-body photo alone.

Why Kling and not Seedance (the Dreamina tab can do motion from @Video1 too):
Seedance refuses reference photos of real human faces, and the whole point of
this tab is putting a real person — usually the user — into the clip.

Tested 16.11.2026: 3s dance + photoreal portrait, with and without the face
element; both accepted, ~7–8 minutes each, 1072x1920 30fps, billed exactly
seconds × rate (2 × 3s Pro = $1.01).
"""

import base64
import io
import os
import re
import tempfile
import time

from PIL import Image, ImageOps

import fal_client
import video_assembler
import clip_studio


class MotionError(Exception):
    pass


# Endpoint schema: https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<id>
# Prices from api.fal.ai/v1/models/pricing (per second of output).
MODELS = {
    "pro": {"id": "fal-ai/kling-video/v3/pro/motion-control",
            "label": "Kling 3.0 Pro", "usd_per_second": 0.168},
    "standard": {"id": "fal-ai/kling-video/v3/standard/motion-control",
                 "label": "Kling 3.0 Standard", "usd_per_second": 0.126},
}
DEFAULT_MODEL = "pro"

# character_orientation: "video" follows the dancer's facing and allows 30s
# (and is the only mode that accepts the face element); "image" keeps the pose
# direction of the photo, follows camera moves better, 10s max.
ORIENTATIONS = {"video": 30, "image": 10}
MIN_SECONDS = 3

IMG_MIN_EDGE = 300          # fal rejects smaller images — at generation time, not upload
IMG_MAX_EDGE = 3000
MAX_FACE_EXTRAS = 3         # reference_image_urls: 1–3

VIDEO_SUBDIR = "motionrefs"
VIDEO_DIR = os.path.join(video_assembler.OUTPUT_DIR, VIDEO_SUBDIR)
VIDEO_MAX_EDGE = 1920


def options():
    return {
        "models": [{"key": k, "label": v["label"], "usd_per_second": v["usd_per_second"]}
                   for k, v in MODELS.items()],
        "default_model": DEFAULT_MODEL,
        "orientations": ORIENTATIONS,
        "min_seconds": MIN_SECONDS,
    }


def estimate_cost(seconds, model=DEFAULT_MODEL):
    spec = MODELS.get(model) or MODELS[DEFAULT_MODEL]
    return round(max(float(seconds or 0), MIN_SECONDS) * spec["usd_per_second"], 2)


# ── Inputs ─────────────────────────────────────────────────────────────────

def _upload(data, filename, content_type):
    """fal storage URL, or a clear error.

    fal_client.upload_bytes falls back to a data: URI when storage refuses —
    fine for a small photo, but a 30s video as base64 inside the JSON request
    is tens of MB. The usual refusal is a locked account (balance exhausted),
    and then the generation would fail anyway, so say it now."""
    url = fal_client.upload_bytes(data, filename, content_type)
    if url.startswith("data:"):
        try:
            bal = fal_client.get_balance()
        except Exception:
            bal = None
        hint = (" Your fal balance is $%.2f — top up at fal.ai/dashboard/billing." % bal
                if isinstance(bal, (int, float)) and bal <= 0 else "")
        raise MotionError("fal refused the upload." + hint)
    return url


def prepare_image(raw):
    """Upload a photo to fal as-is in shape (no crop), sized into fal's limits.

    Returns {"url", "width", "height", "thumb"}."""
    if not raw:
        raise MotionError("That image is empty.")
    try:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise MotionError("That file could not be read as an image. Use JPG, PNG or WEBP.")
    w, h = im.size
    if min(w, h) < IMG_MIN_EDGE:
        s = IMG_MIN_EDGE / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    if max(im.size) > IMG_MAX_EDGE:
        s = IMG_MAX_EDGE / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=93)
    url = _upload(buf.getvalue(), "motion_photo.jpg", "image/jpeg")

    t = im.copy()
    t.thumbnail((240, 240))
    tb = io.BytesIO()
    t.save(tb, "JPEG", quality=80)
    return {"url": url, "width": im.width, "height": im.height,
            "thumb": "data:image/jpeg;base64," + base64.b64encode(tb.getvalue()).decode()}


def _cleanup_videos(max_age=3 * 24 * 3600):
    try:
        now = time.time()
        for n in os.listdir(VIDEO_DIR):
            p = os.path.join(VIDEO_DIR, n)
            if now - os.path.getmtime(p) > max_age:
                os.remove(p)
    except OSError:
        pass


def prepare_video(raw):
    """Re-encode the reference to H.264/AAC mp4 (≤1920 long edge), upload it to fal.

    Returns {"url", "seconds", "width", "height", "thumb", "filename"}."""
    if not raw:
        raise MotionError("That video is empty.")
    if len(raw) > 500 * 1024 * 1024:
        raise MotionError("That video is over 500 MB. Trim it first.")

    os.makedirs(VIDEO_DIR, exist_ok=True)
    _cleanup_videos()
    max_s = max(ORIENTATIONS.values())
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        with open(src, "wb") as f:
            f.write(raw)
        _, info = video_assembler._run(["-i", src], timeout=60)
        m = re.search(r"Stream #\d+:\d+.*?: Video:.*?(\d{2,5})x(\d{2,5})", info)
        seconds = video_assembler._probe_duration(src)
        if not m or not seconds:
            raise MotionError("That file could not be read as a video. Use MP4 or MOV.")
        if seconds < MIN_SECONDS:
            raise MotionError("The reference video must be at least %d seconds; this one is %.1fs."
                              % (MIN_SECONDS, seconds))
        if seconds > max_s + 0.5:
            raise MotionError("The reference video can be at most %d seconds; this one is %.1fs. "
                              "Trim it first." % (max_s, seconds))

        has_audio = bool(re.search(r"Stream #\d+:\d+.*?: Audio:", info))
        name = "motion_%d.mp4" % int(time.time() * 1000)
        dest = os.path.join(VIDEO_DIR, name)
        # -2 keeps both sides even; the min() keeps small clips at their own size.
        scale = ("scale='if(gt(iw,ih),min(%d,iw),-2)':'if(gt(iw,ih),-2,min(%d,ih))'"
                 % (VIDEO_MAX_EDGE, VIDEO_MAX_EDGE))
        args = ["-y", "-i", src, "-t", str(max_s), "-vf", scale,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        args += ["-c:a", "aac", "-b:a", "128k"] if has_audio else ["-an"]
        args += ["-movflags", "+faststart", dest]
        code, err = video_assembler._run(args, timeout=600)
        if code != 0 or not os.path.exists(dest):
            raise MotionError("Could not convert that video: %s" % err.strip()[-300:])
        seconds = round(min(video_assembler._probe_duration(dest) or seconds, max_s), 2)
        _, info2 = video_assembler._run(["-i", dest], timeout=60)
        m2 = re.search(r"Stream #\d+:\d+.*?: Video:.*?(\d{2,5})x(\d{2,5})", info2)
        w, h = (int(m2.group(1)), int(m2.group(2))) if m2 else (int(m.group(1)), int(m.group(2)))

        thumb = ""
        shot = os.path.join(tmp, "thumb.jpg")
        video_assembler._run(["-y", "-ss", str(min(1.0, seconds / 2)), "-i", dest,
                              "-frames:v", "1", "-vf", "scale=-2:160", shot], timeout=60)
        if os.path.exists(shot):
            with open(shot, "rb") as f:
                thumb = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    with open(dest, "rb") as f:
        url = _upload(f.read(), name, "video/mp4")
    return {"url": url, "seconds": seconds, "width": w, "height": h, "thumb": thumb,
            "filename": "%s/%s" % (VIDEO_SUBDIR, name), "has_audio": has_audio}


# ── Generation ─────────────────────────────────────────────────────────────

def normalise(model, orientation, seconds, face_url):
    if model not in MODELS:
        model = DEFAULT_MODEL
    if orientation not in ORIENTATIONS:
        orientation = "video"
    if face_url and orientation != "video":
        # Kling only binds the face element when following the video.
        raise MotionError("The face photo only works with “Follow the video”. "
                          "Switch orientation or remove the face photo.")
    if seconds and float(seconds) > ORIENTATIONS[orientation] + 0.5:
        raise MotionError("“Follow the photo” allows at most %ds of video; yours is %.1fs. "
                          "Use “Follow the video” (up to 30s) or a shorter clip."
                          % (ORIENTATIONS[orientation], float(seconds)))
    return model, orientation


def build_payload(image_url, video_url, orientation, prompt, keep_sound,
                  face_url=None, face_extras=None):
    payload = {"image_url": image_url, "video_url": video_url,
               "character_orientation": orientation,
               "keep_original_sound": bool(keep_sound)}
    prompt = (prompt or "").strip()
    if face_url:
        extras = [u for u in (face_extras or []) if u][:MAX_FACE_EXTRAS] or [image_url]
        payload["elements"] = [{"frontal_image_url": face_url, "reference_image_urls": extras}]
        # The element is only used when the prompt names it.
        if "@element1" not in prompt.lower():
            prompt = ("@Element1 " + prompt).strip() if prompt else "@Element1"
    if prompt:
        payload["prompt"] = prompt
    return payload


def generate_stream(image_url, video_url, seconds, model=DEFAULT_MODEL, orientation="video",
                    prompt="", keep_sound=True, face_url=None, face_extras=None):
    """Make one clip. Yields {"type": "status"|"done"} events."""
    if not image_url:
        yield {"type": "done", "error": "Add the character photo first."}
        return
    if not video_url:
        yield {"type": "done", "error": "Add the reference video first."}
        return
    try:
        model, orientation = normalise(model, orientation, seconds, face_url)
    except MotionError as e:
        yield {"type": "done", "error": str(e)}
        return

    spec = MODELS[model]
    cost = estimate_cost(seconds, model)
    payload = build_payload(image_url, video_url, orientation, prompt, keep_sound,
                            face_url, face_extras)
    try:
        pending = []

        def status(text):
            pending.append(text)

        def drain():
            while pending:
                yield {"type": "status", "text": "   " + pending.pop(0)}

        yield {"type": "status", "text": (
            "🕺 %s · %.1fs · %s · %s · ≈$%.2f" % (
                spec["label"], float(seconds or 0),
                "follow the video" if orientation == "video" else "follow the photo",
                "face locked" if face_url else "no face photo", cost))}
        yield {"type": "status", "text": "⏳ Kling usually takes 5–10 minutes…"}

        out = fal_client.run(spec["id"], payload, on_status=status, timeout=2400)
        yield from drain()

        video = out.get("video") or {}
        url = video.get("url") if isinstance(video, dict) else None
        if not url:
            yield {"type": "done", "error": "The model returned no video: %s" % str(out)[:300]}
            return

        yield {"type": "status", "text": "⬇️ Downloading…"}
        filename = clip_studio.save_output(url, "motion " + (prompt or "transfer"), "mp4")
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename, "video_url": url, "cost": cost,
               "seconds": seconds, "model": model, "model_label": spec["label"],
               "orientation": orientation, "face": bool(face_url),
               "keep_sound": bool(keep_sound), "prompt": payload.get("prompt", "")}
    except fal_client.FalError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": "%s: %s" % (type(e).__name__, e)}
