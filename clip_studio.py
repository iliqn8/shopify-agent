"""Single-clip generator: a prompt, an optional product reference, one video out.

This is the short path next to the Video Cloner. The cloner takes an existing ad
apart and rebuilds it shot by shot; this makes one clip from one prompt, with the
settings exposed directly rather than inferred from a reference.

One thing the models' schemas force, worth knowing before reading the code: on
Seedance's image-to-video endpoint `aspect_ratio` is always "auto" — the shape of
the output comes from the shape of the starting frame, and nothing else. So when
a reference image is supplied, the chosen aspect ratio is applied by cropping
that image before upload. Kling honours the parameter, and text-to-video has no
such constraint either way.

`write_prompt_stream` turns a one-line idea into a prompt for the chosen video
model. It runs on Claude, not fal — the only place in this module that does.
"""

import io
import os
import re
import time
import json
import subprocess

import requests
import anthropic

import fal_client
import video_assembler

# Thinking is on by default on this model, and max_tokens caps thinking plus
# response together, so the budget below is deliberately generous for what is a
# few hundred words of output.
PROMPT_WRITER_MODEL = "claude-opus-5"
claude = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

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


# ── Images for the prompt writer ───────────────────────────────────────────
# These are a different thing from the product reference above. That one is
# uploaded to fal and becomes the clip's first frame; these are shown to Claude
# so it can see what the user means and write the prompt from it. A photo can be
# used for both, but the two paths are separate on purpose.

MAX_IDEA_IMAGES = 4

# Claude Opus 5 accepts up to 2576px on the long edge and downscales anything
# larger itself, so capping here costs no fidelity the model would have used —
# it just avoids pushing megabytes through the browser and the API for pixels
# that get thrown away.
IDEA_IMAGE_MAX_EDGE = 2576
IDEA_THUMB_EDGE = 160


def prepare_idea_image(raw):
    """Ready one uploaded photo for the prompt writer.

    Returns {"b64", "media_type", "thumb"} — the full image for Claude, and a
    small data URI so the browser can show a preview without holding the
    original in memory.
    """
    import cv2
    import numpy as np
    import base64

    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ClipError("That image could not be read. Use a JPEG, PNG or WebP.")

    def encode(img, quality):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ClipError("Could not re-encode that image.")
        return buf.tobytes()

    def fit(img, edge):
        h, w = img.shape[:2]
        if max(h, w) <= edge:
            return img
        scale = edge / max(h, w)
        return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    full = encode(fit(arr, IDEA_IMAGE_MAX_EDGE), 88)
    thumb = encode(fit(arr, IDEA_THUMB_EDGE), 72)
    return {
        "b64": base64.b64encode(full).decode(),
        "media_type": "image/jpeg",
        "thumb": "data:image/jpeg;base64," + base64.b64encode(thumb).decode(),
    }


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


# ── Prompt writing ─────────────────────────────────────────────────────────

# Most of these rules are not general prompt-writing advice — they are the
# failures this project actually paid for in the Video Cloner, written down so
# the same clips don't have to be generated twice.
PROMPT_WRITER_SYSTEM = """You write prompts for image-to-video and text-to-video models. Someone gives you a
rough idea for a product clip; you return the finished prompt they will send to the model.

THE USER'S OWN INSTRUCTIONS COME FIRST — READ THIS BEFORE ANYTHING ELSE

What follows is house style: what to do when the user has described an idea and left the shape of
the prompt to you. It is not a format to impose on someone who has already said what they want.

Users often paste a brief or a template rather than a one-line idea — a structure to follow, a
number of scenes, timestamps, a voiceover per scene, a hook and a call to action, a particular
deliverable at the end. When they do, FOLLOW IT EXACTLY. Produce the number of scenes they ask for,
label them the way they ask, include every element they list, and deliver everything they ask for.
Do not flatten a multi-scene brief into a single paragraph, do not drop the voiceover because the
guidance below is about visuals, and do not silently substitute your own structure for theirs.
Anything below that contradicts what they asked for is overridden by what they asked for.

The rules further down still govern HOW you describe each shot — concrete detail, motion that is
readable, physical grounding. They do not govern how many shots there are or how the output is laid
out. Those are the user's to decide.

WHEN THE SHAPE IS LEFT TO YOU

Write the prompt as plain prose, in English, in the present tense — one paragraph, or two if the
shot genuinely has a second beat. No headings, no bullet lists, no labels like "Camera:", no
key-value pairs. These models read a prompt as a description, not as a form.

WHAT MAKES A PROMPT WORK

Be concrete about what is visible. "A golden retriever shakes water off its coat, droplets catching
the low sun" survives; "a joyful pet moment" does not. Name the subject, the surface it is on, the
light, and the setting in terms someone could photograph.

Describe ONE clear movement. These models degrade badly when asked for several things at once — a
subject action and a camera move is the ceiling for a short clip. Pick the single motion the shot is
about and let everything else hold still.

Give the camera its own clause, and only one instruction: a slow push in, a gentle handheld drift, a
locked-off tripod. If the shot does not need a camera move, say the camera holds still.

END WITH PHYSICAL GROUNDING. This is the rule that matters most and the one people skip. The model
is given a still frame and a sentence; it does not know what is solid. Finish the prompt with a
short sentence covering whichever apply: what carries the subject's weight and that it is rigid,
which objects stay exactly where they are, and where the waterline sits if there is water. Never
write "walks", "steps" or "runs" for movement through water — that phrasing puts the subject on top
of the surface. Say it swims, paddles, or wades with the water at a named height on its body.

Do not ask for ADDED text — captions, subtitles, hook lines burned over the picture, invented
logos. Video models smear lettering into mush, and asking to add or remove a watermark can trip a
provider's content filter on the wording alone. Text that is genuinely part of the scene is a
different matter: a product's own printed label is part of the product, and saying it stays sharp
and readable is fair and often necessary.

LENGTH SHAPES THE CONTENT

A 4-6 second clip holds exactly one beat — one action, start to finish. Around 8-12 seconds you can
carry one action through to a small resolution. Never write more beats than the runtime supports; an
overstuffed prompt makes the model rush everything in it.

Past roughly 15 seconds a single unbroken take usually has nothing left to do, and a sequence of
short scenes is the normal shape — this is how most product ads at that length are built. Give each
scene its own moment and keep each one simple; the per-shot rules above apply to every scene
separately. Bear in mind the model still renders the whole thing in one generation, so cuts come out
as its interpretation of your description rather than as a real edit: make each scene vivid and
distinct enough that the change of shot is unmistakable.

THE FRAMING YOU ARE WRITING FOR

Vertical 9:16 puts the subject close and centred with room above and below; the product should read
at arm's length. Square 1:1 is tight — keep the subject central and lose the wide setting. Landscape
16:9 has room for the subject and its surroundings, so the setting can do real work.

Never mention the aspect ratio, the resolution, the duration, or the model's name inside the prompt
itself. Those are settings, sent separately. Let them shape what you describe, not what you say.

WHEN PHOTOS ARE ATTACHED

The user attaches photos to show you what their words cannot. Read them and write from what you
actually see — the real colour, material, shape, markings, proportions, the light, the place. This
is the whole point of them: "a bright orange foam vest with black webbing straps and a grab handle"
beats "a life jacket", and you can only write the first if you looked.

Describe what is in the photo, not the photograph. The user wants the thing, not its snapshot — so
no "product shot on a white background", no "as pictured", no mention of a studio backdrop or a
phone camera unless the clip is genuinely meant to be set there. If the photo is a plain catalogue
shot of a product, take the product's appearance from it and put that product into the scene the
user described.

The user's words outrank the photos wherever the two disagree. The photos tell you what things look
like; the words tell you what should happen. If they attach a photo of a jacket on a table and ask
for a dog wearing it in a pool, write the pool."""

ANGLE_WRITER_SYSTEM = """You find marketing angles for short product video ads.

An angle is the reason someone watches to the end and then wants the thing — not a feature, and not
a description of the product. "Waterproof, 600D nylon" is a feature. "The first swim after the one
that scared you" is an angle. If your line could be printed on the box, it is not an angle.

Ground every angle in what the page and the photos actually show. Do not invent materials, claims,
certifications, prices or awards; if the source does not say it, do not say it.

Make them genuinely DIFFERENT from one another — not one idea rephrased. Spread them across the
ways people are actually moved to buy: a fear or worry being lifted; a small daily annoyance that
disappears; the moment of transformation before and after; belonging and identity, the kind of owner
this makes you; showing off, how it looks to other people; a specific occasion or season; an
objection answered head on; the product used in a way the buyer had not considered. Not every axis
suits every product — pick the ones that do, and skip the rest rather than forcing a weak angle.

EVERY ANGLE MUST BE FILMABLE AS ONE SHORT CLIP of a few seconds. This is the constraint that makes
an angle useful here rather than just clever. Before you write one, picture the shot: if it needs a
voiceover to make sense, a chart, three cuts, or a testimonial to camera, it does not qualify. Ask
what single visible moment carries the idea, and if there isn't one, drop the angle.

Write for the person choosing between them: a short title they can scan, then the angle itself in
one or two plain sentences, then who it is aimed at. No jargon, no "leverage", no "tap into".

Give between 6 and 9 angles. Fewer good ones beats padding the list."""

ANGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "product": {
            "type": "string",
            "description": ("One short sentence naming what the product actually is, from the page "
                            "and photos. Empty string if you genuinely could not tell."),
        },
        "angles": {
            "type": "array",
            "description": "Between 6 and 9 distinct angles.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string",
                              "description": "3-6 words, scannable in a list."},
                    "angle": {"type": "string",
                              "description": ("The angle itself, one or two plain sentences. This "
                                              "is what gets used to write the video prompt, so make "
                                              "it usable on its own.")},
                    "audience": {"type": "string",
                                 "description": "Who this one is aimed at, a few words."},
                },
                "required": ["title", "angle", "audience"],
                "additionalProperties": False,
            },
        },
        "note": {
            "type": "string",
            "description": ("At most one short sentence if something limited you — e.g. the page "
                            "gave little to work with. Empty string otherwise."),
        },
    },
    "required": ["product", "angles", "note"],
    "additionalProperties": False,
}

REFINE_SYSTEM = PROMPT_WRITER_SYSTEM + """

YOU ARE REVISING A PROMPT THAT ALREADY EXISTS

You are given a prompt you wrote earlier and a change the user wants. Apply that change and return
the whole prompt again.

Change what was asked and leave the rest alone. This is a revision, not a fresh draft: sentences the
request does not touch should come back word for word, so the user can see what moved. Rewriting
details they were happy with is the failure to avoid here.

The rules above still bind the result — one clear movement, the closing physical-grounding sentence,
no on-screen text, no settings named in the prose. If the requested change would break one of them,
make the change in the way that keeps the rule, and say so in your note.

If the request is impossible or would empty the prompt, return the prompt unchanged and explain why
in the note."""

PROMPT_WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": ("The finished prompt, ready to send to the video model — everything "
                            "the video model should read, and nothing else. If the user asked for "
                            "scenes, timestamps or per-scene voiceover lines, they belong here, "
                            "laid out the way they asked."),
        },
        "voiceover_script": {
            "type": "string",
            "description": ("ONLY when the user asked for a separate spoken script as its own "
                            "deliverable. The spoken lines alone, in order, with no labels, "
                            "timestamps or stage directions — ready to paste into a voice tool. "
                            "Empty string whenever they did not ask for one; never duplicate the "
                            "prompt here."),
        },
        "note": {
            "type": "string",
            "description": ("At most one short sentence for the user about a real choice you made "
                            "or a limit they should know — e.g. that their idea has more beats than "
                            "the runtime holds. Empty string if there is nothing worth saying."),
        },
    },
    "required": ["prompt", "voiceover_script", "note"],
    "additionalProperties": False,
}


# Shared by the writer and the reviser so the two cannot drift apart.
_ANGLE_GUIDANCE = (
    "The angle may arrive as up to three parts separated by dashes: a short title, then the angle "
    "itself, then — after \"Aimed at:\" — the audience it is meant for. Read all of it.\n\n"
    "The angle decides WHICH moment you film. Choose the single visible moment that carries it and "
    "build the shot around that, not a generic pretty shot of the product with the angle bolted on. "
    "Do not state the angle in words anywhere in the prompt; a video model cannot film an idea, "
    "only what the idea looks like.\n\n"
    "The audience is WHO THE AD IS FOR. It shapes who appears on camera, the home or place the shot "
    "is set in, the clothes, the time of day, the whole texture of the thing. It is NOT a list of "
    "people to put in the frame: \"people who quit supplements because of pills\" means cast and "
    "stage the shot for that person, not show a group of them."
)


def _writer_brief(idea, model=DEFAULT_MODEL, seconds=8, aspect="16:9",
                  has_image=False, n_photos=0, angle=""):
    spec = model_spec(model)
    angle = (angle or "").strip()
    lines = []
    if n_photos:
        lines += [
            f"The {n_photos} photo(s) above were attached by the user to show you what they mean. "
            "Read them and write from what you actually see in them — real colours, materials, "
            "shapes and markings, not guesses.",
            "",
        ]
    lines += [
        "WHAT THE USER WROTE — their own words, in full:",
        idea.strip() or "(they wrote nothing — work from the photos and the angle)",
        "",
        "If that is a brief or a template rather than a one-line idea — if it asks for a number of "
        "scenes, timestamps, voiceover lines, a hook, a call to action, or a particular deliverable "
        "— then it is an instruction, not a description. Follow it exactly and give them everything "
        "it asks for.",
    ]
    if angle:
        lines += ["", "THE MARKETING ANGLE THIS CLIP MUST SERVE:", angle, "", _ANGLE_GUIDANCE]
    lines += [
        "",
        "THE SETTINGS CURRENTLY SET IN THE APP — this is what the clip will actually be generated at:",
        f"- Video model: {spec['label']}",
        f"- Length: {clamp_seconds(seconds, model)} seconds",
        f"- Framing: {aspect} ({ASPECTS.get(aspect, {}).get('label', '')})",
        "",
        "If what the user wrote asks for a different length or shape from these, write what THEY "
        "asked for and say so in one line in your note, so they can change the setting to match. "
        "Do not quietly cut their brief down to fit a slider they may simply not have moved yet.",
    ]
    if has_image:
        lines += [
            "",
            "SEPARATELY, THEY HAVE SET A PRODUCT REFERENCE THAT BECOMES THE CLIP'S FIRST FRAME. "
            "(It is not shown to you here.) So the subject, the product, the composition and the "
            "setting are already fixed by that frame — do not invent them and do not describe the "
            "product's own appearance, colour or markings, even if a photo above shows them. Write "
            "about what MOVES and what CHANGES from that frame onward: the action, the camera, the "
            "light shifting, what enters or leaves. Where you must refer to the product, do it "
            "generically ('the bottle', 'the jacket') so your words cannot contradict the frame."
            + (" Use the photos above only for the action, mood and setting you are asked to add."
               if n_photos else ""),
        ]
    else:
        lines += [
            "",
            "THERE IS NO FIRST-FRAME REFERENCE — the model builds the whole shot from your words "
            "alone. Describe the subject, the setting and the light as well as the motion, because "
            "nothing else establishes them."
            + (" This is exactly where the attached photos earn their place: take the real "
               "appearance of what they show and put it into your description, so the model renders "
               "the user's actual product rather than a generic stand-in."
               if n_photos else ""),
        ]
    if spec["audio"]["supported"]:
        lines += [
            "",
            "This model can generate synchronised audio. You may name the diegetic sound the scene "
            "would really make (water, footsteps, room tone) in a short clause. No music, and no "
            "dialogue unless the idea calls for someone speaking.",
        ]
    return "\n".join(lines)


def _is_transient(exc):
    """Is this worth trying again in a few seconds?

    Anthropic reports an overload that lands *mid-stream* with the HTTP status
    already sent as 200, so `status_code` says 200 and only the error body says
    what happened — which is how a busy minute produced the useless message
    "Claude API error 200". Match on the error type in the body, then fall back
    to the status code for failures that happen before the stream opens.
    """
    body = str(getattr(exc, "body", "") or "") + " " + str(exc)
    if "overloaded_error" in body or "rate_limit_error" in body:
        return True
    code = getattr(exc, "status_code", None)
    return code in (408, 409, 429, 500, 502, 503, 504, 529)


WRITER_ATTEMPTS = 3


def _ask_claude(system, content, schema, label):
    """One structured call to Claude, retried while it is only busy.

    Generator: yields status events and finally {"data": ...} or {"error": ...},
    so each caller can wrap it in its own event stream without repeating the
    retry, refusal and JSON handling three times over.
    """
    message = None
    for attempt in range(1, WRITER_ATTEMPTS + 1):
        try:
            with claude.messages.stream(
                model=PROMPT_WRITER_MODEL,
                # A multi-scene brief runs to a thousand words of script, and
                # thinking shares this budget — 8000 was sized for a single
                # paragraph and would truncate one of those mid-scene.
                max_tokens=32000,
                system=system,
                output_config={
                    # `high` is the API default and what a plain Claude chat
                    # gets. `medium` was saving tokens at the cost of exactly
                    # the writing quality this feature exists to provide.
                    "effort": "high",
                    "format": {"type": "json_schema", "schema": schema},
                },
                messages=[{"role": "user", "content": content}],
            ) as stream:
                message = stream.get_final_message()
            break
        except anthropic.APIStatusError as e:
            # The SDK retries transport-level 429/5xx itself, but an overload
            # that arrives inside an open stream is not retried — it surfaces
            # here, and it is exactly the case worth repeating.
            if not _is_transient(e):
                yield {"error": f"Claude API error {e.status_code}: {str(e)[:200]}"}
                return
            if attempt == WRITER_ATTEMPTS:
                yield {"error": ("Claude is overloaded at the moment — this is temporary and not "
                                 "a problem with your input. Try again in a minute.")}
                return
            wait = 3 * attempt
            yield {"type": "status", "text": (
                f"⏳ Claude is busy right now — trying again in {wait}s "
                f"(attempt {attempt + 1} of {WRITER_ATTEMPTS})")}
            time.sleep(wait)
        except anthropic.APIConnectionError:
            yield {"error": "Could not reach Claude. Check the network and try again."}
            return

    if message.stop_reason == "refusal":
        yield {"error": "Claude declined this one. Rephrase it and try again."}
        return
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        yield {"data": json.loads(text)}
    except json.JSONDecodeError:
        yield {"error": f"Claude returned something that was not {label}. Try again."}


def _drain(gen):
    """Run one of the helpers above, tagging what comes out.

    Yields ("status", event) for anything the caller should show, then exactly
    one ("data", payload) or ("error", message).
    """
    for item in gen:
        if "error" in item:
            yield ("error", item["error"]); return
        if "data" in item:
            yield ("data", item["data"]); return
        yield ("status", item)
    yield ("error", "Claude produced no result. Try again.")


def _photo_blocks(photos, label="PHOTO"):
    blocks = []
    for i, img in enumerate(photos, start=1):
        blocks.append({"type": "text", "text": f"{label} {i}:"})
        blocks.append({"type": "image", "source": {
            "type": "base64",
            "media_type": img.get("media_type") or "image/jpeg",
            "data": img["b64"],
        }})
    return blocks


def _clean_photos(photos):
    return [p for p in (photos or []) if p and p.get("b64")][:MAX_IDEA_IMAGES]


# ── Marketing angles ───────────────────────────────────────────────────────

def generate_angles_stream(url="", photos=None, note=""):
    """Read a product page (and any photos) and propose angles to choose from."""
    url = (url or "").strip()
    photos = _clean_photos(photos)
    note = (note or "").strip()

    if not url and not photos and not note:
        yield {"type": "done",
               "error": "Paste a product page URL so Claude can see what it is selling."}
        return

    try:
        title, page = None, ""
        if url:
            yield {"type": "status", "text": f"🌐 Reading {url[:70]}…"}
            import section_builder
            title, page = section_builder.fetch_page_text(url)
            # fetch_page_text never raises — it reports failure inside the text.
            if page.startswith("[Could not fetch URL"):
                yield {"type": "status", "text": (
                    "⚠️ That page could not be read (many sites block automated visits) — "
                    "working from the photos and your notes instead")}
                page = ""
            else:
                yield {"type": "status", "text": (
                    f"🌐 Read {len(page.split())} words"
                    + (f' — "{title[:60]}"' if title else ""))}

        content = _photo_blocks(photos, "PRODUCT PHOTO")
        brief = []
        if photos:
            brief += [f"The {len(photos)} photo(s) above show the product. Read them.", ""]
        if url:
            brief.append(f"PRODUCT PAGE: {url}")
            if title:
                brief.append(f"PAGE TITLE: {title}")
            brief += (["", "PAGE TEXT:", page] if page
                      else ["(The page could not be read — work from whatever else is here.)"])
        if note:
            brief += ["", "WHAT THE USER ADDED:", note]
        brief += ["", "Propose the angles."]
        content.append({"type": "text", "text": "\n".join(brief)})

        yield {"type": "status", "text": "💡 Thinking of angles…"}
        for kind, payload in _drain(_ask_claude(
                ANGLE_WRITER_SYSTEM, content, ANGLE_SCHEMA, "a list of angles")):
            if kind == "status":
                yield payload
            elif kind == "error":
                yield {"type": "done", "error": payload}
                return
            else:
                angles = [a for a in (payload.get("angles") or [])
                          if (a.get("angle") or "").strip()]
                if not angles:
                    yield {"type": "done",
                           "error": "Claude found no angles here. Try a different page."}
                    return
                product = (payload.get("product") or "").strip()
                if product:
                    yield {"type": "status", "text": f"📦 {product}"}
                if (payload.get("note") or "").strip():
                    yield {"type": "status", "text": "💡 " + payload["note"].strip()}
                yield {"type": "status", "text": (
                    f"✅ {len(angles)} angle{'s' if len(angles) != 1 else ''} to choose from")}
                yield {"type": "done", "angles": angles, "product": product}
                return

    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


# ── Revising a written prompt ──────────────────────────────────────────────

def refine_prompt_stream(current, instructions, model=DEFAULT_MODEL, seconds=8,
                         aspect="16:9", has_image=False, photos=None, angle=""):
    """Apply a requested change to an existing prompt, keeping the rest intact."""
    current = (current or "").strip()
    instructions = (instructions or "").strip()
    if not current:
        yield {"type": "done", "error": "There is no prompt to change yet."}
        return
    if not instructions:
        yield {"type": "done", "error": "Say what you would like changed."}
        return

    photos = _clean_photos(photos)
    try:
        yield {"type": "status", "text": "🔧 Applying your change…"}

        spec = model_spec(model)
        content = _photo_blocks(photos, "PHOTO FROM THE USER")
        brief = []
        if photos:
            brief += ["The photo(s) above are the user's, for reference.", ""]
        brief += [
            "THE PROMPT AS IT STANDS:",
            current,
            "",
            "THE CHANGE THE USER WANTS:",
            instructions,
            "",
            "THE SETTINGS (unchanged):",
            f"- Video model: {spec['label']}",
            f"- Length: {clamp_seconds(seconds, model)} seconds",
            f"- Framing: {aspect} ({ASPECTS.get(aspect, {}).get('label', '')})",
        ]
        if (angle or "").strip():
            brief += ["", "THE MARKETING ANGLE IT SHOULD STILL SERVE:", angle.strip(),
                      "", _ANGLE_GUIDANCE]
        if has_image:
            brief += ["", "A product reference still supplies the first frame, so keep referring "
                          "to the product generically rather than describing its appearance."]
        content.append({"type": "text", "text": "\n".join(brief)})

        for kind, payload in _drain(_ask_claude(
                REFINE_SYSTEM, content, PROMPT_WRITER_SCHEMA, "a prompt")):
            if kind == "status":
                yield payload
            elif kind == "error":
                yield {"type": "done", "error": payload}
                return
            else:
                prompt = (payload.get("prompt") or "").strip()
                if not prompt:
                    yield {"type": "done", "error": "Claude returned an empty prompt. Try again."}
                    return
                note = (payload.get("note") or "").strip()
                vo = (payload.get("voiceover_script") or "").strip()
                if note:
                    yield {"type": "status", "text": f"💡 {note}"}
                yield {"type": "status", "text": "✅ Updated"}
                yield {"type": "done", "prompt": prompt, "note": note, "voiceover_script": vo}
                return

    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


def _writer_content(idea, model, seconds, aspect, has_image, photos, angle=""):
    """The user turn: the photos first, then the brief.

    Images go before the text they relate to — the model reads the block order,
    and a brief that says "the photos above" has to actually follow them.
    """
    content = _photo_blocks(photos, "PHOTO FROM THE USER")
    content.append({"type": "text", "text": _writer_brief(
        idea, model, seconds, aspect, has_image, n_photos=len(photos), angle=angle)})
    return content


def write_prompt_stream(idea, model=DEFAULT_MODEL, seconds=8, aspect="16:9",
                        has_image=False, photos=None, angle=""):
    """Turn a one-line idea into a finished video prompt. Yields status events.

    `photos` are shown to Claude so it can write from what the user's product
    actually looks like. They are not the clip's first frame — that is the
    separate product reference, and `has_image` says whether one is set.
    `angle` is the marketing angle the clip should serve, if one was chosen.
    """
    idea = (idea or "").strip()
    photos = _clean_photos(photos)
    if not idea and not photos:
        yield {"type": "done", "error": "Describe your idea first, even in one line."}
        return

    try:
        spec = model_spec(model)
        yield {"type": "status", "text": (
            f"✍️ Writing a {clamp_seconds(seconds, model)}s {aspect} prompt for "
            f"{spec['label']}"
            + (f", reading your {len(photos)} photo(s)" if photos else "")
            + (", on your angle" if (angle or "").strip() else "")
            + (" (a first frame is set)" if has_image else "") + "…")}

        for kind, payload in _drain(_ask_claude(
                PROMPT_WRITER_SYSTEM,
                _writer_content(idea, model, seconds, aspect, has_image, photos, angle),
                PROMPT_WRITER_SCHEMA, "a prompt")):
            if kind == "status":
                yield payload
            elif kind == "error":
                yield {"type": "done", "error": payload}
                return
            else:
                prompt = (payload.get("prompt") or "").strip()
                if not prompt:
                    yield {"type": "done", "error": "Claude returned an empty prompt. Try again."}
                    return
                note = (payload.get("note") or "").strip()
                vo = (payload.get("voiceover_script") or "").strip()
                if note:
                    yield {"type": "status", "text": f"💡 {note}"}
                yield {"type": "status", "text": (
                    "✅ Prompt ready" + (" · voiceover script included" if vo else ""))}
                yield {"type": "done", "prompt": prompt, "note": note, "voiceover_script": vo}
                return

    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


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
