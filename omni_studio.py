"""Gemini Omni Flash 1.2 on fal — the Dreamina Studio flow, Google's model.

Same shape as `dreamina_studio`: an idea becomes a prompt, photos and videos go
in as references, one clip comes out, and a finished clip can be edited. The
form, the prompt writer, the AI characters and the queue are shared; what is
different lives here:

  * **Provider.** fal (`google/gemini-omni-flash/v1.2/*`), four endpoints:
    text-to-video, image-to-video, reference-to-video and edit. Picked from what
    the user attached, the same way Dreamina picks a task type.
  * **Shapes.** Only 16:9 and 9:16. **Lengths** 3-15 s. **Resolutions** 360p,
    720p, 1080p and 4k.
  * **Audio is always on** — native, synchronised, no switch in the schema.
  * **References** are up to 10 images and 3 videos (each <= 15 s), addressed in
    the prompt as <IMAGE_REF_0>, <VIDEO_REF_0> (0-based, list order — the
    schema's own example). The UI and the writer keep @Image1/@Video1 like
    Dreamina, and `to_omni_tokens` translates just before sending.
  * **Edit** takes the clip, an instruction and a resolution — no reference
    photos, no length or shape (both come from the source), source <= 15 s.
  * **Price** is per second of output by resolution (fal llms.txt, read
    25.11.2026), plus input tokens at $1.50/M — cents at most, not quoted.

Every file fal needs (data-URI photos, reference videos, the clip to edit) is
uploaded to fal storage first, so nothing depends on this app having a public
address — it works locally too.
"""

import base64
import os
import re

import clip_studio
import dreamina_studio
import fal_client
import video_assembler

ENDPOINTS = {
    "text": "google/gemini-omni-flash/v1.2/text-to-video",
    "image": "google/gemini-omni-flash/v1.2/image-to-video",
    "reference": "google/gemini-omni-flash/v1.2/reference-to-video",
    "edit": "google/gemini-omni-flash/v1.2/edit",
}

MIN_SECONDS = 3
MAX_SECONDS = 15
MAX_REFERENCE_IMAGES = 10
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_VIDEO_SECONDS = 15      # each, not in total
EDIT_MAX_SECONDS = 15

# USD per output second, fal's published rates for v1.2 (all four endpoints).
USD_PER_SECOND = {"360p": 0.0273, "720p": 0.0819, "1080p": 0.1229, "4k": 0.2457}
INPUT_TOKEN_USD_PER_M = 1.50

OMNI_MODELS = {
    "gemini-omni-flash": {
        "label": "Gemini Omni Flash 1.2",
        "tagline": ("3–15s · 360p/720p/1080p/4K · audio always on · "
                    "10 photos + 3 videos as references · from $0.08/s at 720p"),
        "duration": {"mode": "range", "min": MIN_SECONDS, "max": MAX_SECONDS},
        "resolutions": ["360p", "720p", "1080p", "4k"],
        "default_resolution": "720p",
        "aspect_from_image": False,
        "audio": {"supported": True, "free": True, "fixed": True},
        "text_to_video": True,
        "recommended": True,
        "max_images": MAX_REFERENCE_IMAGES,
        "max_videos": MAX_REFERENCE_VIDEOS,
    },
}
DEFAULT_MODEL = "gemini-omni-flash"

ASPECTS = {k: dreamina_studio.ASPECTS[k] for k in ("16:9", "9:16")}

# Pixel sizes are for display only — the price does not depend on them.
DIMENSIONS = {
    "360p": {"16:9": (640, 360), "9:16": (360, 640)},
    "720p": {"16:9": (1280, 720), "9:16": (720, 1280)},
    "1080p": {"16:9": (1920, 1080), "9:16": (1080, 1920)},
    "4k": {"16:9": (3840, 2160), "9:16": (2160, 3840)},
}

RESOLUTION_NOTE = {
    "360p": "Cheapest — for testing an idea",
    "720p": "",
    "1080p": "",
    "4k": "4K — big files, about 3x the 720p price",
}

FORMATS = clip_studio.FORMATS     # fal returns MP4; MOV is clip_studio's remux


class OmniError(Exception):
    pass


def model_spec(key=DEFAULT_MODEL):
    return OMNI_MODELS.get(key) or OMNI_MODELS[DEFAULT_MODEL]


def usd_per_second(resolution="720p"):
    return USD_PER_SECOND.get(resolution) or USD_PER_SECOND["720p"]


def clamp_seconds(seconds, model=DEFAULT_MODEL):
    return clip_studio.clamp_to(model_spec(model), seconds)


def estimate_cost(seconds, resolution="720p", aspect="16:9", model=DEFAULT_MODEL, audio=True):
    return round(usd_per_second(resolution) * clamp_seconds(seconds, model), 4)


def normalise(model=DEFAULT_MODEL, resolution=None, aspect="16:9", container="mp4",
              seconds=8):
    key = model if model in OMNI_MODELS else DEFAULT_MODEL
    spec = OMNI_MODELS[key]
    return {
        "model": key, "spec": spec,
        "resolution": resolution if resolution in spec["resolutions"] else spec["default_resolution"],
        "aspect": aspect if aspect in ASPECTS else "16:9",
        "container": container if container in FORMATS else "mp4",
        "seconds": clip_studio.clamp_to(spec, seconds),
    }


def options():
    models = []
    for key, spec in OMNI_MODELS.items():
        models.append({
            "key": key, "label": spec["label"], "tagline": spec["tagline"],
            "recommended": spec.get("recommended", False),
            "duration": spec["duration"],
            "resolutions": [
                {"key": r, "note": RESOLUTION_NOTE.get(r, ""),
                 "dimensions": {a: "%dx%d" % DIMENSIONS[r][a] for a in ASPECTS},
                 "usd_per_second": {a: usd_per_second(r) for a in ASPECTS}}
                for r in spec["resolutions"]],
            "default_resolution": spec["default_resolution"],
            "flat_usd_per_second": None,
            "audio": spec["audio"],
            "text_to_video": True,
            "aspect_from_image": False,
            "video_usd_per_token": {},
            "max_images": spec["max_images"],
            "max_videos": spec["max_videos"],
        })
    return {
        "models": models,
        "default_model": DEFAULT_MODEL,
        "aspects": [{"key": k, **v} for k, v in ASPECTS.items()],
        "formats": [{"key": k, **v} for k, v in FORMATS.items()],
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
        "promo": None,
        # No `fps` on purpose: the browser only switches to token pricing when
        # it is present. Here a reference video does not change the price.
        "video_input": {
            "max_videos": MAX_REFERENCE_VIDEOS,
            "max_total_seconds": MAX_REFERENCE_VIDEOS * MAX_REFERENCE_VIDEO_SECONDS,
            "max_each_seconds": MAX_REFERENCE_VIDEO_SECONDS,
        },
        "edit_max_seconds": EDIT_MAX_SECONDS,
    }


# ── Reference syntax ───────────────────────────────────────────────────────

def to_omni_tokens(prompt):
    """@Image1 → <IMAGE_REF_0>, @Video2 → <VIDEO_REF_1>. Numbering is 1-based in
    the app (as in Dreamina) and 0-based at Gemini Omni."""
    def img(m):
        return "<IMAGE_REF_%d>" % max(0, int(m.group(1)) - 1)

    def vid(m):
        return "<VIDEO_REF_%d>" % max(0, int(m.group(1)) - 1)

    out = re.sub(r"@\s*[Ii]mage\s*(\d+)", img, prompt or "")
    return re.sub(r"@\s*[Vv]ideo\s*(\d+)", vid, out)


# ── Prompt writing ─────────────────────────────────────────────────────────
PROMPT_CONVENTIONS = """THIS MODEL'S OWN PROMPT CONVENTIONS — Google Gemini Omni Flash:

- Write the prompt as instructions, in this order: subject, action, camera move, look, sound. Start
  with the subject. Leave out any part the idea does not need.
- Camera moves in film language: dolly in, slow push-in, whip pan, snap zoom, 360-degree orbit,
  handheld, locked-off tripod.
- The clip ALWAYS has sound. Spoken lines go inline in double quotes with who says them and how,
  e.g. she says, warmly: "It actually stays cold all day." Keep dialogue short enough to be said
  inside the clip's length.
- End with one sentence listing the sound: effects and ambience, and whether there is music
  ("soft room tone, the click of the lid, no music"). Say "no voice" if nobody should speak.
- Close with what must NOT appear when it matters: "No captions, no on-screen text, no logos."
  This app never wants burnt-in subtitles.
- Write in English. Do not put the aspect ratio, resolution, duration or any other setting into the
  prompt text — they are sent as separate parameters."""

FIRST_FRAME_RULES = """
A STARTING FRAME IS SET: the clip begins on the user's photo. Name the things in that still that must
survive (the product, its colour, its label) and describe only what moves and changes from there."""


def _omni(text):
    return (text.replace("Seedance", "Gemini Omni")
                .replace("the app's aspect setting", "the app's 16:9 / 9:16 setting"))


def _spec_for_writer(model=DEFAULT_MODEL):
    spec = model_spec(model)
    return {"label": spec["label"], "duration": spec["duration"],
            "audio": spec["audio"], "aspects": ASPECTS}


def _extra(has_image, references, references_shown, videos):
    return (PROMPT_CONVENTIONS + (FIRST_FRAME_RULES if has_image else "")
            + _omni(dreamina_studio._reference_rules(references, references_shown))
            + _omni(dreamina_studio._video_rules(videos)))


def write_prompt_stream(idea, model=DEFAULT_MODEL, seconds=8, aspect="16:9",
                        has_image=False, photos=None, angle="", references=0,
                        references_shown=0, videos=0):
    return clip_studio.write_prompt_stream(
        idea, model=model, seconds=seconds, aspect=aspect, has_image=has_image,
        photos=photos, angle=angle, spec=_spec_for_writer(model),
        extra=_extra(has_image, references, references_shown, videos))


def refine_prompt_stream(current, instructions, model=DEFAULT_MODEL, seconds=8,
                         aspect="16:9", has_image=False, photos=None, angle="",
                         references=0, references_shown=0, videos=0):
    return clip_studio.refine_prompt_stream(
        current, instructions, model=model, seconds=seconds, aspect=aspect,
        has_image=has_image, photos=photos, angle=angle, spec=_spec_for_writer(model),
        extra=_extra(has_image, references, references_shown, videos))


# ── Files to fal ───────────────────────────────────────────────────────────

def _to_fal(url, name):
    """A data URI is uploaded to fal storage; an http(s) link is passed as is."""
    if not url:
        return None
    if url.startswith("data:"):
        head, b64 = url.split(",", 1)
        ctype = head[5:].split(";", 1)[0] or "image/jpeg"
        ext = "png" if "png" in ctype else "jpg"
        return fal_client.upload_bytes(base64.b64decode(b64), "%s.%s" % (name, ext), ctype)
    return url


def _local_video(filename):
    """Path of one of our own videos (a finished clip or refvideos/…), or None."""
    name = (filename or "").replace("\\", "/")
    if not name or ".." in name.split("/") or name.startswith("/"):
        return None
    if name.startswith(dreamina_studio.REF_VIDEO_SUBDIR + "/"):
        return dreamina_studio.reference_video_path(name)
    if "/" in name:
        return None
    path = os.path.join(video_assembler.OUTPUT_DIR, name)
    return path if os.path.exists(path) else None


def _upload_video(path):
    with open(path, "rb") as f:
        data = f.read()
    url = fal_client.upload_bytes(data, os.path.basename(path), "video/mp4")
    if url.startswith("data:"):
        raise OmniError("Could not upload the video to fal storage — try again.")
    return url


# ── Generation ─────────────────────────────────────────────────────────────

# fal's Omni endpoints have no negative_prompt field, so "no text" has to be said
# in the prompt itself. Print on the product stays — only added text is banned.
NO_TEXT_LINE = ("No on-screen text of any kind: no captions, subtitles, titles, text overlays, "
                "watermarks, logos or written words added anywhere in the frame. Text printed "
                "on the product itself stays exactly as it is.")


def with_no_text(prompt, no_text):
    prompt = (prompt or "").strip()
    if not no_text or NO_TEXT_LINE in prompt:
        return prompt
    return prompt.rstrip() + ("\n\n" if prompt else "") + NO_TEXT_LINE


def build_payload(prompt, seconds, resolution, aspect, image_url=None,
                  reference_images=None, reference_videos=None, no_text=False):
    """(endpoint, body). Kept apart so it can be tested unpaid."""
    refs = [u for u in (reference_images or []) if u]
    videos = [u for u in (reference_videos or []) if u]
    if image_url and (refs or videos):
        raise OmniError("A starting frame and references cannot be combined — pick one.")
    if len(refs) > MAX_REFERENCE_IMAGES:
        raise OmniError("Gemini Omni takes at most %d reference images; %d were sent."
                        % (MAX_REFERENCE_IMAGES, len(refs)))
    if len(videos) > MAX_REFERENCE_VIDEOS:
        raise OmniError("Gemini Omni takes at most %d reference videos; %d were sent."
                        % (MAX_REFERENCE_VIDEOS, len(videos)))
    body = {"prompt": to_omni_tokens(with_no_text(prompt, no_text)), "aspect_ratio": aspect,
            "resolution": resolution, "duration": int(seconds)}
    if image_url:
        body["image_url"] = image_url
        return ENDPOINTS["image"], body
    if refs or videos:
        if refs:
            body["image_urls"] = refs
        if videos:
            body["reference_video_urls"] = videos
        return ENDPOINTS["reference"], body
    return ENDPOINTS["text"], body


def _events():
    pending = []

    def status(text):
        pending.append(text)

    def drain():
        while pending:
            yield {"type": "status", "text": "   " + pending.pop(0)}
    return status, drain


def generate_stream(prompt, seconds=8, resolution="720p", aspect="16:9", audio=True,
                    container="mp4", image_url=None, reference_images=None,
                    model=DEFAULT_MODEL, reference_videos=None, no_text=False):
    """`reference_videos` is [{"filename", "seconds"}] — our own refvideos/ files."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield {"type": "done", "error": "Write a prompt first — it is the whole instruction."}
        return
    reference_images = [u for u in (reference_images or []) if u]
    reference_videos = [v for v in (reference_videos or []) if v and v.get("filename")]
    for v in reference_videos:
        if float(v.get("seconds") or 0) > MAX_REFERENCE_VIDEO_SECONDS + 0.5:
            yield {"type": "done", "error": (
                "Gemini Omni takes reference videos of at most %d seconds each; one is %.1fs. "
                "Trim it first." % (MAX_REFERENCE_VIDEO_SECONDS, float(v["seconds"])))}
            return

    s = normalise(model, resolution, aspect, container, seconds)
    spec, seconds, resolution, aspect, container = (
        s["spec"], s["seconds"], s["resolution"], s["aspect"], s["container"])
    cost = estimate_cost(seconds, resolution)
    status, drain = _events()

    try:
        w, h = DIMENSIONS[resolution][aspect]
        yield {"type": "status", "text": (
            f"✨ {spec['label']} · {seconds}s · {resolution} {aspect} ({w}×{h}) · with audio · "
            f"{container.upper()} · ≈${cost:.2f}")}

        if reference_images:
            missed = dreamina_studio.unaddressed_references(prompt, len(reference_images))
            if missed:
                yield {"type": "status", "text": "ℹ️ %s not named in the prompt."
                       % ", ".join("@Image%d" % i for i in missed)}
        if reference_videos:
            vmissed = dreamina_studio.unaddressed_videos(prompt, len(reference_videos))
            if vmissed:
                yield {"type": "status", "text": (
                    "ℹ️ %s not named in the prompt — say what to take from it."
                    % ", ".join("@Video%d" % i for i in vmissed))}

        if image_url or reference_images or reference_videos:
            yield {"type": "status", "text": "⬆️ Uploading the references to fal…"}
        first = _to_fal(image_url, "first_frame") if image_url else None
        refs = [_to_fal(u, "ref_%d" % (i + 1)) for i, u in enumerate(reference_images)]
        vids = []
        for v in reference_videos:
            path = _local_video(v["filename"])
            if not path:
                raise OmniError("A reference video is gone from the server (they are kept "
                                "%d days). Upload it again." % dreamina_studio.REF_VIDEO_KEEP_DAYS)
            vids.append(_upload_video(path))

        endpoint, body = build_payload(prompt, seconds, resolution, aspect, first, refs, vids,
                                       no_text=no_text)
        yield {"type": "status", "text": (
            "🖼️ Animating your starting frame…" if first else
            "🖼️ Generating with %d photo%s and %d video%s as references…"
            % (len(refs), "" if len(refs) == 1 else "s", len(vids), "" if len(vids) == 1 else "s")
            if (refs or vids) else "✨ Generating from the prompt alone…")}
        if no_text:
            yield {"type": "status", "text": "   🚫 no on-screen text — added to the prompt"}
        if to_omni_tokens(prompt) != prompt:
            yield {"type": "status", "text": "   references sent as <IMAGE_REF_n> / <VIDEO_REF_n>"}

        out = fal_client.run(endpoint, body, on_status=status, timeout=1800)
        yield from drain()
        url = ((out or {}).get("video") or {}).get("url")
        if not url:
            yield {"type": "done", "error": f"The job finished but returned no video: {str(out)[:300]}"}
            return

        yield {"type": "status", "text": f"⬇️ Downloading the {container.upper()}…"}
        filename = clip_studio.save_output(url, prompt, container)
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename, "video_url": url,
               "cost": cost, "estimated_cost": cost, "tokens": 0,
               "seconds": seconds, "resolution": resolution, "aspect": aspect,
               "audio": True, "container": container, "model": s["model"],
               "model_label": spec["label"], "no_text": bool(no_text)}
    except (fal_client.FalError, OmniError, clip_studio.ClipError) as e:
        yield from drain()
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


# ── Editing a finished clip ────────────────────────────────────────────────

def compose_edit_prompt(instructions, no_text=False):
    """The instruction plus fal's own advice: say what must stay as it is."""
    text = (instructions or "").strip()
    text = re.sub(r"@\s*[Vv]ideo\s*\d*", "the video", text)
    if not re.search(r"\bkeep\b|\bpreserve\b|\bunchanged\b", text, re.I):
        text = text.rstrip(". ") + (". Keep everything else — the people, the product, the "
                                    "lighting, the camera movement and the sound — exactly as it is.")
    return with_no_text(text, no_text)


def edit_stream(filename, instructions, resolution="720p", aspect="16:9",
                container="mp4", source_seconds=None, model=DEFAULT_MODEL, no_text=False):
    instructions = (instructions or "").strip()
    if not instructions:
        yield {"type": "done", "error": "Say what should change about the clip."}
        return
    path = _local_video(filename)
    if not path:
        yield {"type": "done", "error": "That clip is no longer on the server."}
        return
    if source_seconds and float(source_seconds) > EDIT_MAX_SECONDS + 0.5:
        yield {"type": "done", "error": (
            "Gemini Omni can only edit clips up to %d seconds; this one is %ss."
            % (EDIT_MAX_SECONDS, source_seconds))}
        return

    s = normalise(model, resolution, aspect, container, source_seconds or 8)
    spec, resolution, container = s["spec"], s["resolution"], s["container"]
    secs = float(source_seconds or s["seconds"])
    cost = round(usd_per_second(resolution) * secs, 4)
    prompt = compose_edit_prompt(instructions, no_text)
    status, drain = _events()
    try:
        yield {"type": "status", "text": (
            "✏️ Editing the clip · %s · keeps its shape and length · ≈$%.2f" % (resolution, cost))}
        yield {"type": "status", "text": "   sent as: " + prompt}
        yield {"type": "status", "text": "⬆️ Uploading the clip to fal…"}
        video_url = _upload_video(path)
        out = fal_client.run(ENDPOINTS["edit"],
                             {"prompt": prompt, "video_url": video_url, "resolution": resolution},
                             on_status=status, timeout=1800)
        yield from drain()
        url = ((out or {}).get("video") or {}).get("url")
        if not url:
            yield {"type": "done", "error": f"The edit finished but returned no video: {str(out)[:300]}"}
            return
        yield {"type": "status", "text": f"⬇️ Downloading the {container.upper()}…"}
        name = clip_studio.save_output(url, prompt, container)
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": name, "video_url": url, "cost": cost,
               "estimated_cost": cost, "tokens": 0, "seconds": source_seconds,
               "resolution": resolution, "aspect": aspect, "audio": True,
               "container": container, "model": s["model"], "model_label": spec["label"],
               "prompt": prompt, "edited": True, "no_text": bool(no_text)}
    except (fal_client.FalError, OmniError, clip_studio.ClipError) as e:
        yield from drain()
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}
