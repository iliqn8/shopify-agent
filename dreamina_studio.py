"""Dreamina Seedance 2.5 on BytePlus ModelArk — the Clip Studio flow, other rails.

Same shape as `clip_studio`: an idea becomes a prompt, product photos go in, one
clip comes out. What changes is underneath, and the differences are the reason
this is a separate module rather than another entry in CLIP_MODELS:

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
ARK_MIN_RATIO = 0.4
ARK_MAX_RATIO = 2.5
MAX_UPSCALE = 4.0

# Seedance 2.5 takes 1-30 reference images. BytePlus's own prompt guide is
# blunter than the limit: 1-5 works, 6-8 is worth a try but gets unstable. The
# ceiling here is the API's; the warning the UI shows is the guide's.
MAX_REFERENCE_IMAGES = 30
COMFORTABLE_REFERENCES = 5


def _decode(raw):
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise DreaminaError("That image could not be read. Use a JPEG, PNG or WebP.")
    return arr


def _encode(arr):
    import cv2

    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise DreaminaError("Could not re-encode that image.")
    return buf.tobytes(), "image/jpeg"


def _fit_limits(arr, what="image"):
    """Bring an image inside ARK's size window, upscaling or shrinking as needed.

    Shared by both modes because both hit the same wall from opposite sides:
    cropping to a ratio can push the short edge under 300, and a phone photo can
    push the long edge over 6000. Everything travels inline as base64, so the
    upper bound saves request body as well as satisfying the rule.
    """
    import cv2

    h, w = arr.shape[:2]
    short = min(w, h)
    if short < ARK_MIN_EDGE:
        scale = ARK_MIN_EDGE / short
        if scale > MAX_UPSCALE:
            raise DreaminaError(
                "That %s is too small — it is %dx%d, and BytePlus needs at least "
                "%d pixels on the short side. Use a bigger photo."
                % (what, w, h, ARK_MIN_EDGE))
        arr = cv2.resize(arr, (int(math.ceil(w * scale)), int(math.ceil(h * scale))),
                         interpolation=cv2.INTER_CUBIC)

    h, w = arr.shape[:2]
    long_edge = max(w, h)
    if long_edge > ARK_MAX_EDGE:
        scale = ARK_MAX_EDGE / long_edge
        arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return arr


def fit_reference(raw):
    """Prepare a reference image WITHOUT forcing it into the output ratio.

    The difference from `crop_to_aspect` is the whole point of the two modes. A
    first frame becomes the video, so it must be the output's shape. A reference
    image is only material the model reads — cropping it to 9:16 would throw
    away the part of the product that makes it recognisable, for no gain, since
    the output shape comes from `ratio` in this mode.

    The one shape rule that still applies is ARK's own: the image itself must
    sit within [0.4, 2.5]. A panorama or a tall banner gets centre-cropped to
    the nearest edge of that band, and nothing else is touched.
    """
    arr = _decode(raw)
    h, w = arr.shape[:2]
    ratio = w / h

    if ratio > ARK_MAX_RATIO:                 # too wide — trim the sides
        new_w = int(round(h * ARK_MAX_RATIO))
        x = (w - new_w) // 2
        arr = arr[:, x:x + new_w]
    elif ratio < ARK_MIN_RATIO:               # too tall — trim top and bottom
        new_h = int(round(w / ARK_MIN_RATIO))
        y = (h - new_h) // 2
        arr = arr[y:y + new_h]

    return _encode(_fit_limits(arr, "reference image"))


def crop_to_aspect(raw, aspect):
    """Centre-crop image bytes to an aspect ratio. Returns (bytes, content_type).

    This is the ONLY way the chosen ratio reaches the output when a first frame
    is set: Seedance 2.5 rejects any `ratio` but "adaptive" on a first-frame
    task and then copies the shape of this image. Cropping rather than padding,
    because bars baked into the starting frame get animated along with
    everything else.
    """
    arr = _decode(raw)

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
    # floor: 512x512 cropped to 9:16 is 288x512, and 288 is under 300. Scaling
    # back up beats letting the task fail after everything else was right.
    return _encode(_fit_limits(arr, "starting frame"))


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


def prepare_reference(raw, aspect, mode="first_frame"):
    """Return a data URI ARK will accept, prepared for the mode it is used in.

    No upload step: ARK has no storage service, it takes the bytes inline. The
    URI goes back to the browser and is shown as the preview thumbnail, which is
    then literally what the model receives — not a re-render of it.

    `mode` decides whether the chosen ratio is imposed on the image. As a first
    frame it must be, because the output copies that frame's shape; as a
    reference it must not be, because cropping would only destroy detail.
    """
    if mode == "reference":
        data, ctype = fit_reference(raw)
    else:
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


def photos_from_references(reference_images, limit=None):
    """Turn prepared reference data URIs into photos the prompt writer can see.

    The writer has to know what @Image1 actually shows, or it can only name the
    references blindly. They arrive here already prepared for ARK — up to 6000px
    — so each is re-encoded down to the size Claude reads.
    """
    import base64

    limit = clip_studio.MAX_IDEA_IMAGES if limit is None else limit
    out = []
    for uri in (reference_images or [])[:limit]:
        try:
            raw = base64.b64decode((uri or "").split(",", 1)[1])
            out.append(clip_studio.prepare_idea_image(raw))
        except Exception:
            break        # a reference the writer cannot see is not worth failing over
    return out


def _reference_rules(count, shown=0):
    """The house rules for a prompt that has omni reference images behind it.

    Appended to PROMPT_CONVENTIONS rather than folded into the shared writer,
    because addressing references by number is this provider's syntax and would
    be nonsense on fal's models. The insistence on naming them all is about
    control rather than inclusion: an unnamed reference is still read (measured
    - see `unaddressed_references`), but nothing says what it is FOR, and with
    several photos that is how a product ends up borrowing the wrong one's
    colour or setting.

    `shown` is how many of them the writer is actually being shown, and it is
    stated rather than assumed — claiming the photos above are the references
    when they are the user's separate idea photos would have the writer describe
    the wrong pictures.
    """
    if not count:
        return ""
    names = ", ".join("@Image%d" % i for i in range(1, count + 1))
    lines = ["", "THE USER HAS ATTACHED %d REFERENCE IMAGE%s, AND THEY ARE ADDRESSED BY NUMBER:"
             % (count, "" if count == 1 else "S")]
    lines.append("- They are numbered in the order they were uploaded: %s." % names)
    if shown >= count:
        lines.append("- The photos above ARE those references, in that same order: the first photo "
                     "is @Image1, the second @Image2, and so on.")
    elif shown:
        lines.append("- The first %d photo%s above %s @Image1%s, in order. The remaining "
                     "reference%s %s not shown to you — refer to %s by number and keep the wording "
                     "generic." % (shown, "" if shown == 1 else "s",
                                   "is" if shown == 1 else "are",
                                   "" if shown == 1 else "–@Image%d" % shown,
                                   "" if count - shown == 1 else "s",
                                   "is" if count - shown == 1 else "are",
                                   "it" if count - shown == 1 else "them"))
    else:
        lines.append("- You are NOT being shown these references. Name them by number and keep any "
                     "wording about their content generic, so it cannot contradict the photo.")
    lines += [
        "- Seedance reads every reference whether or not you name it, so naming one is how you "
        "say what it is FOR. Write the reference into the sentence where it belongs — \"the bottle "
        "from @Image1 stands on the counter\", \"lit in the style of @Image2\" — rather than "
        "listing them anywhere.",
        "- Name EVERY one of them at least once. An unnamed photo still influences the clip, but "
        "in a way nobody chose. If the idea gives one of them nothing to do, say so in your note "
        "rather than leaving it unmentioned.",
        "- Say what each reference is being taken FOR — the subject's identity, a style, a setting. "
        "The same photo used for \"this exact product\" and \"this general mood\" gives different "
        "results, and the model follows the wording.",
        "- These are NOT first frames. The clip does not have to open on any of them, and the "
        "output shape comes from the app's aspect setting, not from the images.",
    ]
    return "\n".join(lines)


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
                        has_image=False, photos=None, angle="", references=0,
                        references_shown=0):
    return clip_studio.write_prompt_stream(
        idea, model=model, seconds=seconds, aspect=aspect, has_image=has_image,
        photos=photos, angle=angle, spec=_spec_for_writer(model),
        extra=PROMPT_CONVENTIONS + _reference_rules(references, references_shown))


def refine_prompt_stream(current, instructions, model=DEFAULT_MODEL, seconds=8,
                         aspect="16:9", has_image=False, photos=None, angle="",
                         references=0, references_shown=0):
    return clip_studio.refine_prompt_stream(
        current, instructions, model=model, seconds=seconds, aspect=aspect,
        has_image=has_image, photos=photos, angle=angle,
        spec=_spec_for_writer(model),
        extra=PROMPT_CONVENTIONS + _reference_rules(references, references_shown))


# Finding a marketing angle is about the product, not the video model.
generate_angles_stream = clip_studio.generate_angles_stream


# ── Generation ─────────────────────────────────────────────────────────────

def build_payload(spec, prompt, seconds, resolution, aspect, audio, container,
                  image_url=None, reference_images=None):
    """The ARK request body. Kept apart so it can be tested unpaid.

    There are two mutually exclusive ways to give this model a photo, and mixing
    them is rejected by the API, not merely ignored:

      * **First frame** (`role: first_frame`, exactly one image). The clip
        literally begins on that frame, so `ratio` MUST be "adaptive" — any
        other value is refused — and the output copies the image's shape. The
        chosen ratio has already been applied by cropping that image.
      * **Omni references** (`role: reference_image`, 1-30 images). The images
        are material the model reads rather than a frame it starts from, so
        `ratio` and `duration` behave normally and no cropping is needed. The
        catch is that references are addressed FROM THE PROMPT, as @Image1,
        @Image2 in the order sent here. A reference nobody names in the prompt
        is loose material the model may or may not use.

    `omni_reference_task_type` is sent explicitly as "reference" rather than
    left at "auto" because auto defers the decision to a worker: a prompt that
    reads like an edit ("replace the bottle") would be reclassified after the
    task was accepted and fail asynchronously, minutes later. Declared up front,
    the same mistake comes back from the submit call while the user is watching.

    `duration` is a number, not a string, and ARK validates it strictly when
    parameters are sent in the body rather than as `--flags` on the prompt. The
    body is the documented way; the flag form is legacy and silently ignores
    what it does not understand.
    """
    refs = [u for u in (reference_images or []) if u]
    if refs and image_url:
        raise DreaminaError(
            "A starting frame and reference images cannot be combined — Seedance "
            "treats them as different kinds of task. Pick one.")
    if len(refs) > MAX_REFERENCE_IMAGES:
        raise DreaminaError(
            "Seedance 2.5 takes at most %d reference images; %d were sent."
            % (MAX_REFERENCE_IMAGES, len(refs)))

    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url},
            "role": "first_frame",
        })
    for url in refs:
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
            "role": "reference_image",
        })

    payload = {
        "model": spec["id"],
        "content": content,
        "resolution": resolution,
        "ratio": "adaptive" if image_url else aspect,
        "duration": int(seconds),
        "generate_audio": bool(audio),
        "output_format": container,
        "watermark": False,
    }
    if refs:
        payload["omni_reference_task_type"] = "reference"
    return payload


def unaddressed_references(prompt, count):
    """Which of the N reference images the prompt never names. 1-based.

    Measured, not assumed. Two references were sent with a prompt that named
    neither, and both were still read — colours that appear nowhere in the text
    came through into the clip. So an unnamed reference is NOT ignored.

    What naming buys is direction. Unnamed, the model decides for itself what
    each photo contributes; @Image1 is how the prompt says which photo is the
    product, which is the setting, which is only the mood. With one obvious
    reference that hardly matters, and BytePlus's own examples still name them.
    """
    named = {int(n) for n in re.findall(r"@\s*[Ii]mage\s*(\d+)", prompt or "")}
    return [i for i in range(1, count + 1) if i not in named]


# ── Editing a finished clip ────────────────────────────────────────────────
# Seedance's third task type. It re-generates the clip with a change applied,
# rather than patching pixels, so the result is a NEW video of roughly the same
# length — BytePlus say it can come back up to 0.4s shorter, and at a
# non-integer duration.
#
# The constraints are not advice. `ratio` must be adaptive, `duration` must be
# -1, the source must be 4-30 seconds, and the prompt must contain an editing
# verb or the model will classify the task as something else and fail it.
EDIT_MIN_SECONDS = 4
EDIT_MAX_SECONDS = 30

# An edit is billed at roughly TWICE a generation of the same shape and length —
# the source video is read as well as the result written. Measured, not guessed:
# a 7.04s 480p 9:16 edit billed 134,905 tokens where the published formula gives
# 67,636 for that output. Quoting one of those and charging the other is the kind
# of surprise this app exists to avoid.
EDIT_BILLING_MULTIPLIER = 2.0

# BytePlus's own list, plus the obvious synonyms a person actually types.
EDIT_TRIGGERS = ("edit", "add", "insert", "remove", "delete", "modify",
                 "replace", "change", "swap", "erase", "take out", "put")


def estimate_edit_cost(seconds, resolution="720p", aspect="16:9",
                       model=DEFAULT_MODEL):
    """What an edit of a clip this long will cost. See EDIT_BILLING_MULTIPLIER."""
    return round(estimate_cost(seconds, resolution, aspect, model)
                 * EDIT_BILLING_MULTIPLIER, 4)


def has_edit_trigger(text):
    low = " %s " % (text or "").lower()
    return any(t in low for t in EDIT_TRIGGERS)


def compose_edit_prompt(instructions):
    """Turn what the user typed into something Seedance will read as an edit.

    Two things have to be true of the text, and neither is worth making the user
    remember: it must name the asset being edited (@Video1), and it must contain
    an editing verb. BytePlus's own example is shaped exactly like the prefix
    below — "Video edit: remove everyone in @Video1 except the protagonist."
    """
    text = (instructions or "").strip()
    if not text:
        return ""
    if re.search(r"@\s*[Vv]ideo\s*\d*", text):
        # They addressed the clip themselves; only guarantee the verb.
        return text if has_edit_trigger(text) else "Video edit: " + text
    return "Video edit on @Video1: " + text


def build_edit_payload(spec, prompt, video_url, resolution, audio, container,
                       reference_images=None):
    """The ARK body for an edit. Separate from build_payload because almost
    every field is pinned differently, and sharing one function would mean a
    branch on every line.
    """
    refs = [u for u in (reference_images or []) if u]
    if len(refs) > MAX_REFERENCE_IMAGES:
        raise DreaminaError(
            "Seedance 2.5 takes at most %d reference images; %d were sent."
            % (MAX_REFERENCE_IMAGES, len(refs)))
    if not video_url:
        raise DreaminaError("There is no video to edit.")

    content = [
        {"type": "text", "text": prompt},
        {"type": "video_url", "video_url": {"url": video_url},
         "role": "reference_video"},
    ]
    for url in refs:
        content.append({"type": "image_url", "image_url": {"url": url},
                        "role": "reference_image"})

    return {
        "model": spec["id"],
        "content": content,
        "resolution": resolution,
        "ratio": "adaptive",          # pinned by the API for edits
        "duration": -1,               # likewise: the source's length is kept
        "generate_audio": bool(audio),
        "output_format": container,
        "watermark": False,
        "omni_reference_task_type": "edit",
    }


def edit_stream(video_url, instructions, reference_images=None,
                resolution="720p", aspect="16:9", audio=True, container="mp4",
                source_seconds=None, model=DEFAULT_MODEL):
    """Re-generate a finished clip with a change applied.

    `video_url` must be reachable from the internet — ARK fetches it, and unlike
    images there is no base64 form for video, so a local file cannot be sent.
    """
    instructions = (instructions or "").strip()
    if not instructions:
        yield {"type": "done", "error": "Say what should change about the clip."}
        return
    if not video_url:
        yield {"type": "done",
               "error": "This clip has no address the model can fetch it from."}
        return

    reference_images = [u for u in (reference_images or []) if u]
    s = normalise(model=model, resolution=resolution, aspect=aspect, audio=audio,
                  container=container, seconds=source_seconds or EDIT_MIN_SECONDS)
    spec = s["spec"]
    resolution, container, audio = s["resolution"], s["container"], s["audio"]

    if source_seconds and not (EDIT_MIN_SECONDS <= source_seconds <= EDIT_MAX_SECONDS):
        yield {"type": "done", "error": (
            "Seedance can only edit a clip between %d and %d seconds; this one is %ss."
            % (EDIT_MIN_SECONDS, EDIT_MAX_SECONDS, source_seconds))}
        return

    prompt = compose_edit_prompt(instructions)
    # The edit keeps the source's length, and is billed at about double a
    # generation of that length, because the source is read as well as written.
    cost = estimate_edit_cost(source_seconds or EDIT_MIN_SECONDS, resolution,
                              aspect, s["model"])

    try:
        pending = []

        def status(text):
            pending.append(text)

        def drain():
            while pending:
                yield {"type": "status", "text": "   " + pending.pop(0)}

        yield {"type": "status", "text": (
            "✏️ Editing the clip · %s · keeps its %s shape and length · ≈$%.2f"
            % (resolution, aspect, cost))}
        if prompt != instructions:
            yield {"type": "status", "text": "   sent as: " + prompt}
        if reference_images:
            yield {"type": "status", "text": (
                "   with %d reference photo%s — @Image1%s"
                % (len(reference_images), "" if len(reference_images) == 1 else "s",
                   "–@Image%d" % len(reference_images) if len(reference_images) > 1 else ""))}

        payload = build_edit_payload(spec, prompt, video_url, resolution, audio,
                                     container, reference_images)
        task = byteplus_client.run(payload, on_status=status, timeout=1800)
        yield from drain()

        url = ((task.get("content") or {}).get("video_url"))
        if not url:
            yield {"type": "done",
                   "error": f"The edit finished but returned no video: {str(task)[:300]}"}
            return

        made_seconds = task.get("duration") or source_seconds
        made_resolution = task.get("resolution") or resolution
        tokens = (task.get("usage") or {}).get("completion_tokens") or 0
        actual = cost_of_tokens(tokens, made_resolution) if tokens else cost

        yield {"type": "status", "text": f"⬇️ Downloading the {container.upper()}…"}
        filename = save_output(url, prompt, container)

        if tokens:
            yield {"type": "status", "text": f"💰 Billed {tokens:,} tokens ≈ ${actual:.2f}"}
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename, "video_url": url,
               "cost": actual, "estimated_cost": cost, "tokens": tokens,
               "seconds": made_seconds, "resolution": made_resolution,
               "aspect": task.get("ratio") or aspect, "audio": bool(audio),
               "container": container, "model": s["model"],
               "model_label": spec["label"], "prompt": prompt, "edited": True}

    except byteplus_client.ArkError as e:
        msg = str(e)
        if "TaskTypeMismatch" in msg or "TaskTypeConstraint" in msg:
            msg += ("  —  Seedance read this as something other than an edit. Phrase it as a "
                    "change to the existing clip (\"remove the…\", \"replace the… with…\").")
        yield {"type": "done", "error": msg}
    except DreaminaError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


def generate_stream(prompt, seconds=8, resolution="720p", aspect="16:9",
                    audio=True, container="mp4", image_url=None,
                    reference_images=None, model=DEFAULT_MODEL):
    """Make one clip on BytePlus. Yields {"type": "status"|"done"} events."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield {"type": "done", "error": "Write a prompt first — it is the whole instruction."}
        return
    reference_images = [u for u in (reference_images or []) if u]

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

        if reference_images:
            missed = unaddressed_references(prompt, len(reference_images))
            if missed:
                yield {"type": "status", "text": (
                    "ℹ️ %s not named in the prompt — still read, but the model "
                    "decides what %s for."
                    % (", ".join("@Image%d" % i for i in missed),
                       "it is used" if len(missed) == 1 else "they are used"))}

        payload = build_payload(spec, prompt, seconds, resolution, aspect, audio,
                                container, image_url, reference_images)
        yield {"type": "status", "text": (
            "🖼️ Animating your reference image — the clip takes its shape from that frame…"
            if image_url else
            "🖼️ Generating with %d reference image%s — @Image1%s…"
            % (len(reference_images), "" if len(reference_images) == 1 else "s",
               "–@Image%d" % len(reference_images) if len(reference_images) > 1 else "")
            if reference_images else "✨ Generating from the prompt alone…")}

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
