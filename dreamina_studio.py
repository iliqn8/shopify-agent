"""Dreamina Seedance 2.5 on BytePlus ModelArk — the Clip Studio flow, other rails.

Same shape as `clip_studio`: an idea becomes a prompt, an optional product photo
becomes the first frame, one clip comes out. What changes is underneath, and the
differences are the reason this is a separate module rather than another entry in
CLIP_MODELS:

  * **Provider.** BytePlus ModelArk, not fal. One task endpoint, polled; images
    go inline as data URIs because ARK has no file storage.
  * **Price.** Seedance 2.5 direct from BytePlus is roughly half what the same
    model costs through fal — $0.231/s at 720p against about $0.50/s.
  * **Shapes.** Six aspect ratios, not three: 21:9, 4:3 and 3:4 as well as the
    usual landscape, square and vertical.
  * **A hard constraint fal does not have.** When a first-frame image is sent,
    Seedance 2.5 REFUSES any `ratio` other than "adaptive" and takes the output
    shape from that image. So the ratio the user picks is applied by cropping the
    reference before it is sent — exactly the trick Clip Studio uses on Seedance,
    but here it is enforced by the API rather than merely ignored.
  * **Containers are native.** `output_format` gives mp4 or mov straight from
    the model, so there is no ffmpeg remux step at all.

The prompt writer is Claude, shared with `clip_studio` rather than copied — the
craft of writing a shot does not change with the provider. What is passed in is
this model's spec and its own prompt conventions (`PROMPT_CONVENTIONS`), which
are real and specific: Seedance 2.5 reads bracket markup for sound.
"""

import os
import re
import time
import math

import requests

import clip_studio
import byteplus_client
import video_assembler

# One model today, and the registry shape is kept anyway: BytePlus also sells
# Seedance 2.0, 2.0 fast, 2.0 mini and 1.5 pro on the same endpoint, and adding
# one is an entry here plus its price row. Only what has been verified against
# the published spec belongs in it.
MODEL_ID = "dreamina-seedance-2-5-260628"

MIN_SECONDS = 4
MAX_SECONDS = 30

DREAMINA_MODELS = {
    "seedance-2.5": {
        "id": MODEL_ID,
        "label": "Dreamina Seedance 2.5",
        "tagline": ("4–30s · 480p/720p/1080p · free audio · six shapes · "
                    "about half the fal price"),
        "duration": {"mode": "range", "min": MIN_SECONDS, "max": MAX_SECONDS},
        "resolutions": ["480p", "720p", "1080p"],
        "default_resolution": "720p",
        # Not a preference — the API rejects any other ratio when a first-frame
        # image is present, so the reference is cropped instead.
        "aspect_from_image": True,
        "audio": {"supported": True, "free": True},
        "text_to_video": True,
        "recommended": True,
    },
}

DEFAULT_MODEL = "seedance-2.5"


def model_spec(key=DEFAULT_MODEL):
    return DREAMINA_MODELS.get(key) or DREAMINA_MODELS[DEFAULT_MODEL]


class DreaminaError(Exception):
    pass


# ── Shapes and sizes ───────────────────────────────────────────────────────
# Read off the BytePlus "width and height pixel values corresponding to
# different aspect ratios" table, Seedance 2.5 column. These are not derived
# from the ratio: 720p 4:3 is 1112x834, which is neither 960x720 nor anything a
# formula would produce. They matter because the price is computed from the
# pixel count, so guessing them would misquote the bill.
DIMENSIONS = {
    "480p": {
        "16:9": (854, 480), "9:16": (480, 854), "1:1": (640, 640),
        "4:3": (752, 560), "3:4": (560, 752), "21:9": (992, 432),
    },
    "720p": {
        "16:9": (1280, 720), "9:16": (720, 1280), "1:1": (960, 960),
        "4:3": (1112, 834), "3:4": (834, 1112), "21:9": (1470, 630),
    },
    "1080p": {
        "16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1440, 1440),
        "4:3": (1664, 1248), "3:4": (1248, 1664), "21:9": (2206, 946),
    },
}

ASPECTS = {
    "16:9": {"label": "Landscape", "w": 16, "h": 9,
             "note": "YouTube, website hero, Facebook feed"},
    "1:1": {"label": "Square", "w": 1, "h": 1,
            "note": "Instagram feed, product grids"},
    "9:16": {"label": "Vertical", "w": 9, "h": 16,
             "note": "Reels, TikTok, Stories"},
    "4:3": {"label": "Classic", "w": 4, "h": 3,
            "note": "Slides, older displays, a softer landscape"},
    "3:4": {"label": "Portrait", "w": 3, "h": 4,
            "note": "Pinterest, print-shaped product shots"},
    "21:9": {"label": "Cinematic", "w": 21, "h": 9,
             "note": "Widescreen banners, a film look"},
}

FORMATS = {
    "mp4": {"label": "MP4", "note": "Universal — web, ads, every platform"},
    "mov": {"label": "MOV",
            "note": "Higher colour precision for Premiere / Final Cut — some players cannot open it"},
}

RESOLUTION_NOTE = {
    "480p": "8-bit colour",
    "720p": "8-bit colour",
    "1080p": "10-bit colour, H.265 — beautiful, but VLC or a modern player to view it",
}


# ── Price ──────────────────────────────────────────────────────────────────
# BytePlus bills video by token, where
#
#     tokens = duration x frame rate x width x height / 1024
#
# at a fixed 24 fps. The per-token rate below is derived from BytePlus's own
# published price examples for Seedance 2.5 at 16:9, 5 seconds:
#
#     480p   $0.514 / 48,038 tokens  = 1.0700e-5
#     720p   $1.156 / 108,000 tokens = 1.0704e-5
#     1080p  $2.843 / 243,000 tokens = 1.1700e-5
#
# 480p and 720p agreeing to four figures is what makes this trustworthy rather
# than a guess; 1080p really is dearer per token, which fits it being the only
# tier rendered at 10-bit.
FPS = 24
USD_PER_TOKEN = {"480p": 1.0700e-5, "720p": 1.0704e-5, "1080p": 1.1700e-5}

# A live BytePlus promotion: 1080p at 28% off, ending 14:00 on 17 September 2026
# in UTC+8, which is 06:00 UTC. Encoded with its expiry rather than baked into
# the rate, so the quote goes back up by itself instead of quietly understating
# the bill from the 18th onward.
PROMO = {
    "resolutions": ("1080p",),
    "multiplier": 0.72,
    "until": 1789279200,          # 2026-09-17 06:00 UTC
    "label": "1080p is 28% off at BytePlus until 17 Sept",
}


def promo_active(now=None):
    return (now or time.time()) < PROMO["until"]


def dimensions(resolution, aspect):
    table = DIMENSIONS.get(resolution) or DIMENSIONS["720p"]
    return table.get(aspect) or table["16:9"]


def usd_per_second(resolution="720p", aspect="16:9", now=None):
    """Price of one generated second at this shape. A rate, not a clip."""
    w, h = dimensions(resolution, aspect)
    rate = USD_PER_TOKEN.get(resolution, USD_PER_TOKEN["720p"])
    if resolution in PROMO["resolutions"] and promo_active(now):
        rate *= PROMO["multiplier"]
    return round(w * h * FPS / 1024 * rate, 4)


def clamp_seconds(seconds, model=DEFAULT_MODEL):
    return clip_studio.clamp_to(model_spec(model), seconds)


def estimate_cost(seconds, resolution="720p", aspect="16:9",
                  model=DEFAULT_MODEL, audio=True):
    """What this clip will cost, at the length the model will actually make.

    `audio` is accepted and ignored on purpose: Seedance 2.5 charges the same
    with sound as without, and the callers are shaped like Clip Studio's, where
    audio does move the price on the Kling models.
    """
    return round(usd_per_second(resolution, aspect)
                 * clamp_seconds(seconds, model), 4)


def cost_of_tokens(tokens, resolution="720p", now=None):
    """The real bill, from the token count BytePlus returns on a finished task.

    This is the number that was actually charged. The estimate above is the
    published formula, and the two differ by about a frame's worth because the
    model renders duration x 24 + 1 frames — a 12s clip came back as 289.
    """
    rate = USD_PER_TOKEN.get(resolution, USD_PER_TOKEN["720p"])
    if resolution in PROMO["resolutions"] and promo_active(now):
        rate *= PROMO["multiplier"]
    return round((tokens or 0) * rate, 4)


def normalise(model=DEFAULT_MODEL, resolution=None, aspect="16:9", audio=True,
              container="mp4", seconds=MIN_SECONDS, has_image=False):
    """Settle every setting against what Seedance 2.5 actually accepts.

    One place, used by both the estimate and the generation, so the price quoted
    is the price of the clip that gets made.
    """
    key = model if model in DREAMINA_MODELS else DEFAULT_MODEL
    spec = DREAMINA_MODELS[key]
    return {
        "model": key,
        "spec": spec,
        "resolution": resolution if resolution in spec["resolutions"]
                      else spec["default_resolution"],
        "aspect": aspect if aspect in ASPECTS else "16:9",
        "audio": bool(audio),
        "container": container if container in FORMATS else "mp4",
        "seconds": clip_studio.clamp_to(spec, seconds),
    }


def options():
    """Everything the UI needs to draw the form, priced from one place."""
    models = []
    for key, spec in DREAMINA_MODELS.items():
        models.append({
            "key": key,
            "label": spec["label"],
            "tagline": spec["tagline"],
            "recommended": spec.get("recommended", False),
            "duration": spec["duration"],
            "resolutions": [
                {"key": r,
                 "note": RESOLUTION_NOTE.get(r, ""),
                 "dimensions": {a: "%dx%d" % dimensions(r, a) for a in ASPECTS},
                 "usd_per_second": {a: usd_per_second(r, a) for a in ASPECTS}}
                for r in spec["resolutions"]
            ],
            "default_resolution": spec["default_resolution"],
            "flat_usd_per_second": None,
            "audio": spec["audio"],
            "text_to_video": spec["text_to_video"],
            "aspect_from_image": spec["aspect_from_image"],
        })

    return {
        "models": models,
        "default_model": DEFAULT_MODEL,
        "aspects": [{"key": k, **v} for k, v in ASPECTS.items()],
        "formats": [{"key": k, **v} for k, v in FORMATS.items()],
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
        "promo": PROMO["label"] if promo_active() else None,
    }


# ── Reference image ────────────────────────────────────────────────────────
# BytePlus wants both edges in [300, 6000] and the ratio inside [0.4, 2.5].
# Every ratio offered here sits inside that band — 21:9 is 2.33, 3:4 is 0.75 —
# so cropping to the chosen ratio can only fail on size, never on shape.
ARK_MIN_EDGE = 300
ARK_MAX_EDGE = 6000
MAX_UPSCALE = 4.0


def crop_to_aspect(raw, aspect):
    """Centre-crop image bytes to an aspect ratio. Returns (bytes, content_type).

    This is the ONLY way the chosen ratio reaches the output when a first frame
    is set: Seedance 2.5 rejects any `ratio` but "adaptive" on a first-frame
    task and then copies the shape of this image. Cropping rather than padding,
    because bars baked into the starting frame get animated along with
    everything else.
    """
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise DreaminaError("That image could not be read. Use a JPEG, PNG or WebP.")

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

    # Cropping only ever removes pixels, and an ordinary photo falls through the
    # floor: 512x512 cropped to 9:16 is 288x512, and 288 is under 300. Scale it
    # back up rather than letting the task fail after everything else was right.
    h2, w2 = arr.shape[:2]
    short = min(w2, h2)
    if short < ARK_MIN_EDGE:
        scale = ARK_MIN_EDGE / short
        if scale > MAX_UPSCALE:
            raise DreaminaError(
                "That image is too small to use as a starting frame. After cropping to "
                "%s it is %dx%d, and BytePlus needs at least %dx%d. Use a photo at "
                "least %d pixels on its short side."
                % (aspect, w2, h2, ARK_MIN_EDGE, ARK_MIN_EDGE,
                   int(math.ceil(ARK_MIN_EDGE / MAX_UPSCALE * (max(w2, h2) / short)))))
        arr = cv2.resize(arr, (int(math.ceil(w2 * scale)), int(math.ceil(h2 * scale))),
                         interpolation=cv2.INTER_CUBIC)

    # The other end of the same rule. A phone photo cropped to 21:9 can pass
    # 6000 on the long edge, and the whole thing travels inline as base64, so
    # shrinking here saves the request body as well as satisfying the limit.
    h3, w3 = arr.shape[:2]
    long_edge = max(w3, h3)
    if long_edge > ARK_MAX_EDGE:
        scale = ARK_MAX_EDGE / long_edge
        arr = cv2.resize(arr, (max(1, int(w3 * scale)), max(1, int(h3 * scale))),
                         interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise DreaminaError("Could not re-encode that image.")
    return buf.tobytes(), "image/jpeg"


def fetch_reference(url, timeout=60):
    """Download a reference image from a URL the user pasted."""
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ShopifyAgent/1.0)"})
        r.raise_for_status()
    except Exception as e:
        raise DreaminaError(f"Could not fetch that URL: {e}")
    if not r.content:
        raise DreaminaError("That URL returned an empty file.")
    ctype = (r.headers.get("Content-Type") or "").lower()
    if ctype and not ctype.startswith("image/"):
        raise DreaminaError(
            f"That URL is {ctype.split(';')[0]}, not an image. Link straight to the "
            "image file, not to the page it sits on.")
    return r.content


def prepare_reference(raw, aspect):
    """Crop to the chosen ratio and return a data URI ARK will accept.

    No upload step: ARK has no storage service, it takes the bytes inline. The
    URI goes back to the browser and is shown as the preview thumbnail, which is
    then literally the frame the model starts from — not a re-render of it.
    """
    data, ctype = crop_to_aspect(raw, aspect)
    return byteplus_client.to_data_uri(data, ctype)


# Photos for the prompt writer are provider-neutral — Claude reads them either
# way — so this is the same function Clip Studio uses, not a copy of it.
prepare_idea_image = clip_studio.prepare_idea_image
MAX_IDEA_IMAGES = clip_studio.MAX_IDEA_IMAGES


# ── Output ─────────────────────────────────────────────────────────────────

def save_output(video_url, prompt, container="mp4", output_dir=None):
    """Download the finished clip.

    No conversion: `output_format` was sent with the task, so BytePlus already
    rendered the container that was asked for. Clip Studio has to remux to MOV
    with ffmpeg because fal only ever returns MP4; here the file that arrives is
    the file that was ordered.
    """
    output_dir = output_dir or video_assembler.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    raw = byteplus_client.download(video_url)
    words = re.findall(r"[a-z0-9]+", (prompt or "clip").lower())[:5]
    stem = "dreamina_%s_%d" % ("_".join(words) or "clip", int(time.time()))
    name = stem + ("." + container if container in FORMATS else ".mp4")
    with open(os.path.join(output_dir, name), "wb") as f:
        f.write(raw)
    return name


# ── Prompt writing ─────────────────────────────────────────────────────────
# Seedance 2.5's own conventions, from the BytePlus prompt guide. These are real
# syntax the model reads, not style advice, which is why they are handed to the
# writer as house rules rather than left to it to invent.
PROMPT_CONVENTIONS = """THIS MODEL'S OWN PROMPT CONVENTIONS — Dreamina Seedance 2.5 reads these as syntax:

- Order the description as subject + action + scene and environment + visual style + camera
  movement + sound. Leave out any part the idea does not need.
- Sound is written with brackets, and the brackets mean different things:
    (…) music        <…> sound effects        {…} spoken dialogue        【…】 on-screen subtitles
  Use them only for sound the clip should actually contain. Never use 【…】 — this app does not
  want burnt-in subtitles.
- Dialogue in {…} gets lip-synced, so keep it to what can be said inside the clip's length.
- The model understands English, Spanish, Portuguese, Japanese, Korean, Arabic, Thai, Vietnamese,
  Indonesian and Malay. Write the prompt in English; only put another language inside {…} if the
  user asked for the spoken line to be in it, and name the language just before the braces.
- Do not put the aspect ratio, resolution, duration or any other setting into the prompt text. They
  are sent as separate parameters."""


def _spec_for_writer(model=DEFAULT_MODEL):
    """The shape `clip_studio`'s writer expects, describing this model."""
    spec = model_spec(model)
    return {
        "label": spec["label"],
        "duration": spec["duration"],
        "audio": spec["audio"],
        "aspects": ASPECTS,
    }


def write_prompt_stream(idea, model=DEFAULT_MODEL, seconds=8, aspect="16:9",
                        has_image=False, photos=None, angle=""):
    return clip_studio.write_prompt_stream(
        idea, model=model, seconds=seconds, aspect=aspect, has_image=has_image,
        photos=photos, angle=angle,
        spec=_spec_for_writer(model), extra=PROMPT_CONVENTIONS)


def refine_prompt_stream(current, instructions, model=DEFAULT_MODEL, seconds=8,
                         aspect="16:9", has_image=False, photos=None, angle=""):
    return clip_studio.refine_prompt_stream(
        current, instructions, model=model, seconds=seconds, aspect=aspect,
        has_image=has_image, photos=photos, angle=angle,
        spec=_spec_for_writer(model), extra=PROMPT_CONVENTIONS)


# Finding a marketing angle is about the product, not the video model.
generate_angles_stream = clip_studio.generate_angles_stream


# ── Generation ─────────────────────────────────────────────────────────────

def build_payload(spec, prompt, seconds, resolution, aspect, audio, container,
                  image_url=None):
    """The ARK request body. Kept apart so it can be tested unpaid.

    Two things here are constraints, not choices:

      * With a first-frame image, `ratio` MUST be "adaptive" — anything else is
        rejected outright, and the output takes the shape of the image. The
        chosen ratio has already been applied by cropping that image.
      * `duration` is a number, not a string, and ARK validates it strictly when
        parameters are sent in the body rather than as `--flags` on the prompt.
        The body is the documented way; the flag form is the legacy one and
        silently ignores what it does not understand.
    """
    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url},
            "role": "first_frame",
        })

    return {
        "model": spec["id"],
        "content": content,
        "resolution": resolution,
        "ratio": "adaptive" if image_url else aspect,
        "duration": int(seconds),
        "generate_audio": bool(audio),
        "output_format": container,
        "watermark": False,
    }


def generate_stream(prompt, seconds=8, resolution="720p", aspect="16:9",
                    audio=True, container="mp4", image_url=None,
                    model=DEFAULT_MODEL):
    """Make one clip on BytePlus. Yields {"type": "status"|"done"} events."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield {"type": "done", "error": "Write a prompt first — it is the whole instruction."}
        return

    s = normalise(model=model, resolution=resolution, aspect=aspect, audio=audio,
                  container=container, seconds=seconds, has_image=bool(image_url))
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

        w, h = dimensions(resolution, aspect)
        yield {"type": "status", "text": (
            f"🌙 {spec['label']} · {seconds}s · {resolution} {aspect} ({w}×{h}) · "
            f"{'with audio' if audio else 'silent'} · {container.upper()} · ≈${cost:.2f}")}

        payload = build_payload(spec, prompt, seconds, resolution, aspect, audio,
                                container, image_url)
        yield {"type": "status", "text": (
            "🖼️ Animating your reference image — the clip takes its shape from that frame…"
            if image_url else "✨ Generating from the prompt alone…")}

        task = byteplus_client.run(payload, on_status=status, timeout=1800)
        yield from drain()

        url = ((task.get("content") or {}).get("video_url"))
        if not url:
            yield {"type": "done",
                   "error": f"The task finished but returned no video: {str(task)[:300]}"}
            return

        # What the model actually made, which is not always what was ordered:
        # duration comes back rounded down from the real frame count, and the
        # token count is the number BytePlus bills on.
        made_seconds = task.get("duration") or seconds
        made_resolution = task.get("resolution") or resolution
        tokens = (task.get("usage") or {}).get("completion_tokens") or 0
        actual = cost_of_tokens(tokens, made_resolution) if tokens else cost

        yield {"type": "status", "text": f"⬇️ Downloading the {container.upper()}…"}
        filename = save_output(url, prompt, container)

        yield {"type": "status", "text": (
            f"💰 Billed {tokens:,} tokens ≈ ${actual:.2f}" if tokens else "✅ Done")}
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename, "video_url": url,
               "cost": actual, "estimated_cost": cost, "tokens": tokens,
               "seconds": made_seconds, "resolution": made_resolution,
               "aspect": task.get("ratio") or aspect, "audio": bool(audio),
               "container": container, "model": s["model"],
               "model_label": spec["label"]}

    except byteplus_client.ArkError as e:
        yield {"type": "done", "error": str(e)}
    except DreaminaError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}
