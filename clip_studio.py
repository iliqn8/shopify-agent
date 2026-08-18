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

MIN_SECONDS = 4
MAX_SECONDS = 30

# ── Model registry ─────────────────────────────────────────────────────────
# Read off each endpoint's own schema, not assumed. The models genuinely differ
# in what they will accept, and sending a field a model does not have is not an
# error on fal — it is accepted and silently ignored, so a wrong entry here
# looks like the model misbehaving rather than like a bug.
#
#   duration      "range" takes any whole second in [min,max]; "choice" sells
#                 only the listed lengths and anything else is rounded to one.
#   resolutions   None means the endpoint has no resolution field at all.
#   aspect_from_image
#                 True when the image-to-video endpoint pins aspect_ratio to
#                 "auto" and takes the shape from the starting frame. The
#                 reference is cropped either way; this only decides whether the
#                 parameter is sent.
#   audio         supported=False hides the control; free=False means the rate
#                 doubles with audio on, which is real money on Kling.
#   pricing       "tokens" bills by pixel count (see USD_PER_1K_TOKENS);
#                 "flat" is a published per-second rate.
CLIP_MODELS = {
    "seedance-2.5": {
        "label": "Seedance 2.5",
        "tagline": "Any length 4–30s · 480p/720p/1080p · free audio · sharpest",
        "i2v": "bytedance/seedance-2.5/image-to-video",
        "t2v": "bytedance/seedance-2.5/text-to-video",
        "image_field": "image_url",
        "duration": {"mode": "range", "min": 4, "max": 30},
        "resolutions": ["480p", "720p", "1080p"],
        "default_resolution": "720p",
        "aspect_from_image": True,
        "audio": {"supported": True, "free": True},
        "pricing": {"kind": "tokens", "per_1k": {"480p": 0.0214, "720p": 0.0214, "1080p": 0.0234}},
        "negative_prompt": False,
        "recommended": True,
    },
    "kling-2.6-pro": {
        "label": "Kling 2.6 Pro",
        "tagline": "5s or 10s · most realistic per dollar · audio doubles the rate",
        "i2v": "fal-ai/kling-video/v2.6/pro/image-to-video",
        "t2v": "fal-ai/kling-video/v2.6/pro/text-to-video",
        # 2.6 renamed this field. Sending `image_url` is accepted and ignored,
        # which quietly downgrades the call to text-to-video.
        "image_field": "start_image_url",
        "duration": {"mode": "choice", "values": [5, 10]},
        "resolutions": None,
        "aspect_from_image": False,
        "audio": {"supported": True, "free": False},
        "pricing": {"kind": "flat", "usd_per_second": 0.07, "audio_multiplier": 2.0},
        "negative_prompt": True,
    },
    "kling-2.5-pro": {
        "label": "Kling 2.5 Turbo Pro",
        "tagline": "5s or 10s · same price as 2.6 · no audio on this endpoint",
        "i2v": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
        "t2v": "fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
        "image_field": "image_url",
        "duration": {"mode": "choice", "values": [5, 10]},
        "resolutions": None,
        "aspect_from_image": False,
        "audio": {"supported": False, "free": True},
        "pricing": {"kind": "flat", "usd_per_second": 0.07},
        "negative_prompt": True,
    },
    "seedance-2.0": {
        "label": "Seedance 2.0",
        "tagline": "4–12s · finest detail · four times Kling",
        "i2v": "bytedance/seedance-2.0/image-to-video",
        "t2v": None,
        "image_field": "image_url",
        "duration": {"mode": "choice", "values": [4, 6, 8, 10, 12]},
        "resolutions": ["480p", "720p", "1080p"],
        "default_resolution": "720p",
        "aspect_from_image": True,
        "audio": {"supported": False, "free": True},
        # $0.3034/s at 720p 16:9 works back to this rate through the same
        # token formula, so 1080p scales correctly instead of being guessed.
        "pricing": {"kind": "tokens", "per_1k": {"480p": 0.01404, "720p": 0.01404, "1080p": 0.01404}},
        "negative_prompt": False,
    },
    "kling-2.5-standard": {
        "label": "Kling 2.5 Standard",
        "tagline": "5s or 10s · cheapest · visibly weaker, good for drafts",
        "i2v": "fal-ai/kling-video/v2.5-turbo/standard/image-to-video",
        "t2v": "fal-ai/kling-video/v2.5-turbo/standard/text-to-video",
        "image_field": "image_url",
        "duration": {"mode": "choice", "values": [5, 10]},
        "resolutions": None,
        "aspect_from_image": False,
        "audio": {"supported": False, "free": True},
        "pricing": {"kind": "flat", "usd_per_second": 0.042},
        "negative_prompt": True,
    },
}

DEFAULT_MODEL = "seedance-2.5"


def model_spec(key):
    return CLIP_MODELS.get(key) or CLIP_MODELS[DEFAULT_MODEL]

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

# The token-priced models bill tokens = height * width * seconds * 24 / 1024,
# times a per-1000 rate that lives in each model's `pricing` entry. fal's
# published per-second figures for Seedance 2.5 at 16:9 (~$0.2205 at 480p,
# ~$0.4730 at 720p) come out a few percent above what the formula gives for
# exactly 854x480 and 1280x720, so they round the frame up somewhere. The
# estimate carries that margin rather than understating the bill.
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


def usd_per_second(model=DEFAULT_MODEL, resolution="720p", aspect="16:9", audio=True):
    """Price of one generated second. Not clamped — a rate, not a clip."""
    spec = model_spec(model)
    pricing = spec["pricing"]
    if pricing["kind"] == "flat":
        rate = pricing["usd_per_second"]
        if audio and spec["audio"]["supported"] and not spec["audio"]["free"]:
            rate *= pricing.get("audio_multiplier", 1.0)
        return round(rate, 4)

    w, h = dimensions(resolution, aspect)
    per_1k = pricing["per_1k"].get(resolution, list(pricing["per_1k"].values())[0])
    return round((w * h * 24) / 1024 / 1000 * per_1k * ESTIMATE_MARGIN, 4)


def clamp_seconds(seconds, model=DEFAULT_MODEL):
    """The length this model will actually sell, nearest to what was asked.

    Models that sell fixed blocks are the reason this exists: asking Kling for
    23 seconds and being handed 10 without a word is exactly the kind of silent
    mismatch that makes a clip look broken rather than mispriced.
    """
    spec = model_spec(model)
    try:
        want = float(seconds)
    except (TypeError, ValueError):
        want = MIN_SECONDS

    d = spec["duration"]
    if d["mode"] == "choice":
        return min(d["values"], key=lambda v: (abs(v - want), v))
    return max(d["min"], min(d["max"], int(round(want))))


def estimate_cost(seconds, resolution="720p", aspect="16:9",
                  model=DEFAULT_MODEL, audio=True):
    """What this clip will cost, in USD, at the length the model will sell.

    Keep this separate from `usd_per_second`: this one clamps, and multiplying a
    clamped 1 by a per-second rate is not the same thing as the rate itself.
    """
    return round(usd_per_second(model, resolution, aspect, audio)
                 * clamp_seconds(seconds, model), 4)


def normalise(model=DEFAULT_MODEL, resolution=None, aspect="16:9",
              audio=True, container="mp4", seconds=MIN_SECONDS, has_image=False):
    """Settle every setting against what the chosen model actually supports.

    One place, used by both the estimate and the generation, so the price quoted
    is the price of the clip that gets made.
    """
    key = model if model in CLIP_MODELS else DEFAULT_MODEL
    spec = CLIP_MODELS[key]

    if spec["resolutions"]:
        resolution = resolution if resolution in spec["resolutions"] else \
            spec.get("default_resolution", spec["resolutions"][0])
    else:
        resolution = None

    audio = bool(audio) and spec["audio"]["supported"]
    if not has_image and not spec["t2v"]:
        # Nothing to animate and no text-to-video endpoint to fall back on.
        raise ClipError(
            f"{spec['label']} can only animate an image. Add a product photo or an "
            "image URL, or pick a model that generates from text alone.")

    return {
        "model": key,
        "spec": spec,
        "resolution": resolution,
        "aspect": aspect if aspect in ASPECTS else "16:9",
        "audio": audio,
        "container": container if container in FORMATS else "mp4",
        "seconds": clamp_seconds(seconds, key),
    }


def options():
    """Everything the UI needs to draw the form, priced from one place.

    Each model carries its own capabilities so the controls can follow the
    choice: a model that sells only 5s and 10s should not be offered a slider,
    and one with no resolution field should not be shown resolutions.
    """
    models = []
    for key, spec in CLIP_MODELS.items():
        res = spec["resolutions"]
        models.append({
            "key": key,
            "label": spec["label"],
            "tagline": spec["tagline"],
            "recommended": spec.get("recommended", False),
            "duration": spec["duration"],
            "resolutions": [
                {"key": r,
                 "dimensions": {a: "%dx%d" % dimensions(r, a) for a in ASPECTS},
                 "usd_per_second": {a: usd_per_second(key, r, a, True) for a in ASPECTS}}
                for r in (res or [])
            ],
            "default_resolution": spec.get("default_resolution"),
            # With no resolution field the rate is flat, so the UI still needs a
            # number to multiply by. Quote it both ways where audio moves it.
            "flat_usd_per_second": None if res else {
                "audio": usd_per_second(key, None, "16:9", True),
                "silent": usd_per_second(key, None, "16:9", False),
            },
            "audio": spec["audio"],
            "text_to_video": bool(spec["t2v"]),
            "aspect_from_image": spec["aspect_from_image"],
        })
    # Deliberate order, not alphabetical: best first, draft-quality last.
    # Sorting by label put "Kling 2.5 Standard — visibly weaker" second.
    rank = {k: i for i, k in enumerate(
        ("seedance-2.5", "kling-2.6-pro", "kling-2.5-pro",
         "seedance-2.0", "kling-2.5-standard"))}
    models.sort(key=lambda m: rank.get(m["key"], 99))

    return {
        "models": models,
        "default_model": DEFAULT_MODEL,
        "aspects": [{"key": k, **v} for k, v in ASPECTS.items()],
        "formats": [{"key": k, **v} for k, v in FORMATS.items()],
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

NEGATIVE_PROMPT = "blur, distort, low quality, text, watermark, warped hands"


def build_payload(spec, prompt, seconds, resolution, aspect, audio, image_url):
    """The request body for one model. Kept apart so it can be tested unpaid.

    Every difference here came from that endpoint's own schema. Sending a field
    a model does not have is not rejected by fal — it is ignored — so a mistake
    here is invisible until the output is wrong.
    """
    payload = {"prompt": prompt}

    if image_url:
        payload[spec["image_field"]] = image_url

    # Kling wants the length as a string; Seedance wants a number.
    payload["duration"] = str(seconds) if spec["pricing"]["kind"] == "flat" else int(seconds)

    if spec["resolutions"] and resolution:
        payload["resolution"] = resolution

    # Only send the ratio where the endpoint honours it. Seedance's
    # image-to-video pins it to "auto" and takes the shape from the frame, which
    # is why the reference is cropped before upload either way.
    if not (image_url and spec["aspect_from_image"]):
        payload["aspect_ratio"] = aspect

    if spec["audio"]["supported"]:
        payload["generate_audio"] = bool(audio)

    if spec["negative_prompt"]:
        payload["negative_prompt"] = NEGATIVE_PROMPT

    return payload


def generate_stream(prompt, seconds=8, resolution="720p", aspect="16:9",
                    audio=True, container="mp4", image_url=None,
                    model=DEFAULT_MODEL):
    """Make one clip. Yields {"type": "status"|"done"} events."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield {"type": "done", "error": "Write a prompt first — it is the whole instruction."}
        return

    try:
        s = normalise(model=model, resolution=resolution, aspect=aspect, audio=audio,
                      container=container, seconds=seconds, has_image=bool(image_url))
    except ClipError as e:
        yield {"type": "done", "error": str(e)}
        return

    spec = s["spec"]
    seconds, resolution, aspect = s["seconds"], s["resolution"], s["aspect"]
    audio, container = s["audio"], s["container"]
    cost = estimate_cost(seconds, resolution, aspect, s["model"], audio)

    try:
        pending = []

        def status(text):
            pending.append(text)

        def drain():
            while pending:
                yield {"type": "status", "text": "   " + pending.pop(0)}

        shape = f"{resolution} {aspect} (%d×%d)" % dimensions(resolution, aspect) \
            if resolution else aspect
        yield {"type": "status", "text": (
            f"🎞️ {spec['label']} · {seconds}s · {shape} · "
            f"{'with audio' if audio else 'silent'} · ≈${cost:.2f}")}

        model_id = spec["i2v"] if image_url else spec["t2v"]
        payload = build_payload(spec, prompt, seconds, resolution, aspect, audio, image_url)
        yield {"type": "status", "text": (
            "🖼️ Animating your reference image…" if image_url
            else "✨ Generating from the prompt alone…")}

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
               "aspect": aspect, "audio": bool(audio), "container": container,
               "model": s["model"], "model_label": spec["label"]}

    except fal_client.FalError as e:
        yield {"type": "done", "error": str(e)}
    except ClipError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}
