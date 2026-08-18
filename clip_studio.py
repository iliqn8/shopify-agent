"""Single-clip generator: a prompt, an optional product reference, one video out.

This is the short path next to the Video Cloner. The cloner takes an existing ad
apart and rebuilds it shot by shot; this makes one clip from one prompt, with the
settings exposed directly rather than inferred from a reference.

Everything runs on Seedance 2.5, the only model in the registry that takes an
arbitrary whole-second length (4-30) and can return synchronised audio, which is
what this tab is for.

One thing the model's schema forces, worth knowing before reading the code: on
the image-to-video endpoint `aspect_ratio` is always "auto" — the shape of the
output comes from the shape of the starting frame, and nothing else. So when a
reference image is supplied, the chosen aspect ratio is applied by cropping that
image before upload. Text-to-video has no such constraint and takes the ratio
directly.
"""

import io
import os
import re
import time
import subprocess

import requests

import fal_client
import video_assembler

I2V_MODEL = "bytedance/seedance-2.5/image-to-video"
T2V_MODEL = "bytedance/seedance-2.5/text-to-video"

MIN_SECONDS = 4
MAX_SECONDS = 30

# A resolution label fixes the PIXEL COUNT, not the short edge. Asking for 480p
# at 1:1 returned a 640x640 clip, not 480x480 — and 640x640 is 409,600 pixels
# against 854x480's 409,920. So the square is the same size to render and the
# same price, and treating it as a 480-tall frame understated the bill by 44%.
SHORT_EDGE = {"480p": 480, "720p": 720, "1080p": 1080}
LONG_EDGE = {"480p": 854, "720p": 1280, "1080p": 1920}

ASPECTS = {
    "16:9": {"label": "Landscape", "w": 16, "h": 9,
             "note": "YouTube, website hero, Facebook feed"},
    "1:1": {"label": "Square", "w": 1, "h": 1,
            "note": "Instagram feed, product grids"},
    "9:16": {"label": "Vertical", "w": 9, "h": 16,
             "note": "Reels, TikTok, Stories"},
}

FORMATS = {
    "mp4": {"label": "MP4", "note": "Universal — web, ads, every platform"},
    "mov": {"label": "MOV", "note": "For Premiere / Final Cut / After Effects"},
}

# fal bills tokens = height * width * seconds * 24 / 1024, at $0.0214 per 1000
# tokens for 480p and 720p and $0.0234 for 1080p. Their published per-second
# figures for 16:9 (~$0.2205 at 480p, ~$0.4730 at 720p) come out a few percent
# above what the formula gives for exactly 854x480 and 1280x720, so they round
# the frame up somewhere. The estimate carries that margin rather than
# understating the bill.
USD_PER_1K_TOKENS = {"480p": 0.0214, "720p": 0.0214, "1080p": 0.0234}
ESTIMATE_MARGIN = 1.08


class ClipError(Exception):
    pass


def dimensions(resolution, aspect):
    """Output pixel size for a resolution label and an aspect ratio.

    The square case is derived from the 16:9 pixel budget rather than assumed:
    a real 480p 1:1 generation came back 640x640, which is that budget to within
    0.1%, so the rule is "same number of pixels, different shape".
    """
    short = SHORT_EDGE.get(resolution, 720)
    long_edge = LONG_EDGE.get(resolution, 1280)
    if aspect == "1:1":
        side = int(round((short * long_edge) ** 0.5))
        return side, side
    if aspect == "9:16":
        return short, long_edge          # w, h
    return long_edge, short              # 16:9


def usd_per_second(resolution="720p", aspect="16:9"):
    """Price of one generated second in this shape. Not clamped — a rate, not a clip."""
    w, h = dimensions(resolution, aspect)
    rate = USD_PER_1K_TOKENS.get(resolution, 0.0214)
    return round((w * h * 24) / 1024 / 1000 * rate * ESTIMATE_MARGIN, 4)


def clamp_seconds(seconds):
    try:
        seconds = int(round(float(seconds)))
    except (TypeError, ValueError):
        seconds = MIN_SECONDS
    return max(MIN_SECONDS, min(MAX_SECONDS, seconds))


def estimate_cost(seconds, resolution="720p", aspect="16:9"):
    """What this clip will cost, in USD. Same formula fal bills on.

    Keep this separate from `usd_per_second`: this one clamps to a length the
    model will actually sell, and multiplying a clamped 1 by a per-second rate
    is not the same thing as the rate itself.
    """
    return round(usd_per_second(resolution, aspect) * clamp_seconds(seconds), 4)


def options():
    """Everything the UI needs to draw the form, priced from one place."""
    return {
        "aspects": [{"key": k, **v} for k, v in ASPECTS.items()],
        "formats": [{"key": k, **v} for k, v in FORMATS.items()],
        "resolutions": [
            {"key": r,
             "label": r,
             "dimensions": {a: "%dx%d" % dimensions(r, a) for a in ASPECTS},
             # Per shape, because billing is by pixel count: a square 1080p
             # frame is genuinely cheaper than a 16:9 one with the same label.
             "usd_per_second": {a: usd_per_second(r, a) for a in ASPECTS}}
            for r in ("480p", "720p", "1080p")
        ],
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
    }


# ── Reference image ────────────────────────────────────────────────────────

def crop_to_aspect(raw, aspect):
    """Centre-crop image bytes to an aspect ratio. Returns (bytes, content_type).

    The image-to-video endpoint ignores `aspect_ratio` entirely and takes the
    output shape from this frame, so cropping here is the only way the chosen
    ratio can be honoured. Cropping rather than padding: bars baked into the
    starting frame get animated along with everything else.
    """
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ClipError("That image could not be read. Use a JPEG, PNG or WebP.")

    h, w = arr.shape[:2]
    spec = ASPECTS.get(aspect) or ASPECTS["16:9"]
    want = spec["w"] / spec["h"]
    have = w / h

    if abs(have - want) > 0.01:
        if have > want:                       # too wide — trim the sides
            new_w = int(round(h * want))
            x = (w - new_w) // 2
            arr = arr[:, x:x + new_w]
        else:                                 # too tall — trim top and bottom
            new_h = int(round(w / want))
            y = (h - new_h) // 2
            arr = arr[y:y + new_h]

    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ClipError("Could not re-encode that image.")
    return buf.tobytes(), "image/jpeg"


def fetch_reference(url, timeout=60):
    """Download a reference image from a URL the user pasted."""
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ShopifyAgent/1.0)"})
        r.raise_for_status()
    except Exception as e:
        raise ClipError(f"Could not fetch that URL: {e}")
    if not r.content:
        raise ClipError("That URL returned an empty file.")
    ctype = (r.headers.get("Content-Type") or "").lower()
    if ctype and not ctype.startswith("image/"):
        raise ClipError(
            f"That URL is {ctype.split(';')[0]}, not an image. Link straight to the "
            "image file, not to the page it sits on.")
    return r.content


def prepare_reference(raw, aspect):
    """Crop to the chosen ratio and upload. Returns a fal URL."""
    data, ctype = crop_to_aspect(raw, aspect)
    return fal_client.upload_bytes(data, "clip_reference.jpg", ctype)


# ── Output ─────────────────────────────────────────────────────────────────

def _safe_stem(prompt):
    words = re.findall(r"[a-z0-9]+", (prompt or "clip").lower())[:5]
    return "_".join(words) or "clip"


def save_output(video_url, prompt, container="mp4", output_dir=None):
    """Download the clip and write it out in the requested container.

    MOV is a remux, not a re-encode: the H.264/AAC streams fal returns are legal
    in a QuickTime container, so `-c copy` changes the wrapper in under a second
    and loses nothing. Re-encoding here would cost quality for no reason.
    """
    output_dir = output_dir or video_assembler.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    raw = fal_client.download(video_url)
    stem = f"clip_{_safe_stem(prompt)}_{int(time.time())}"
    src = os.path.join(output_dir, stem + ".mp4")
    with open(src, "wb") as f:
        f.write(raw)

    if container != "mov":
        return os.path.basename(src)

    dst = os.path.join(output_dir, stem + ".mov")
    proc = subprocess.run(
        [video_assembler.ffmpeg_exe(), "-y", "-i", src,
         "-c", "copy", "-movflags", "+faststart", dst],
        capture_output=True, timeout=600)
    if proc.returncode != 0 or not os.path.exists(dst):
        # Keep the mp4 rather than failing the whole run over a container swap.
        raise ClipError(
            "MOV conversion failed, the MP4 is still here: "
            + proc.stderr.decode("utf-8", "replace")[-300:])
    try:
        os.remove(src)
    except OSError:
        pass
    return os.path.basename(dst)


# ── Generation ─────────────────────────────────────────────────────────────

def generate_stream(prompt, seconds=8, resolution="720p", aspect="16:9",
                    audio=True, container="mp4", image_url=None):
    """Make one clip. Yields {"type": "status"|"done"} events."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield {"type": "done", "error": "Write a prompt first — it is the whole instruction."}
        return

    seconds = clamp_seconds(seconds)
    resolution = resolution if resolution in SHORT_EDGE else "720p"
    aspect = aspect if aspect in ASPECTS else "16:9"
    container = container if container in FORMATS else "mp4"

    w, h = dimensions(resolution, aspect)
    cost = estimate_cost(seconds, resolution, aspect)

    try:
        pending = []

        def status(text):
            pending.append(text)

        def drain():
            while pending:
                yield {"type": "status", "text": "   " + pending.pop(0)}

        yield {"type": "status", "text": (
            f"🎞️ {seconds}s · {resolution} {aspect} ({w}×{h}) · "
            f"{'with audio' if audio else 'silent'} · ≈${cost:.2f}")}

        if image_url:
            # aspect_ratio is not a parameter here — the reference frame was
            # cropped to the chosen ratio on the way in, and the output follows it.
            model_id = I2V_MODEL
            payload = {
                "prompt": prompt,
                "image_url": image_url,
                "duration": seconds,
                "resolution": resolution,
                "generate_audio": bool(audio),
            }
            yield {"type": "status", "text": "🖼️ Animating your reference image…"}
        else:
            model_id = T2V_MODEL
            payload = {
                "prompt": prompt,
                "duration": seconds,
                "resolution": resolution,
                "aspect_ratio": aspect,
                "generate_audio": bool(audio),
            }
            yield {"type": "status", "text": "✨ Generating from the prompt alone…"}

        out = fal_client.run(model_id, payload, on_status=status, timeout=1800)
        yield from drain()

        video = (out.get("video") or {})
        url = video.get("url") if isinstance(video, dict) else None
        if not url:
            yield {"type": "done", "error": f"The model returned no video: {str(out)[:300]}"}
            return

        yield {"type": "status", "text": f"⬇️ Downloading and packaging as {container.upper()}…"}
        filename = save_output(url, prompt, container)

        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename, "video_url": url,
               "cost": cost, "seconds": seconds, "resolution": resolution,
               "aspect": aspect, "audio": bool(audio), "container": container}

    except fal_client.FalError as e:
        yield {"type": "done", "error": str(e)}
    except ClipError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}
