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
MIN_PAD_FROM = 2.5          # 2.5–3 s videos are padded to 3 s instead of refused

IMG_MIN_EDGE = 300          # fal rejects smaller images — at generation time, not upload
IMG_MAX_EDGE = 3000
MAX_FACE_EXTRAS = 3         # reference_image_urls: 1–3

VIDEO_SUBDIR = "motionrefs"
VIDEO_DIR = os.path.join(video_assembler.OUTPUT_DIR, VIDEO_SUBDIR)
VIDEO_MAX_EDGE = 1920

# ── Objects swap: Kling O3 Edit ─────────────────────────────────────────────
# Keeps the video (motion, camera, timing) and replaces named things in it with
# the things in the photos — a product, a prop, a person, several at once.
# Schema: video 3–15 s, mp4/mov, 720–3840 px; `elements` → @Element1..N;
# elements + image_urls ≤ 4 in total. Price checked 17.11.2026 on
# api.fal.ai/v1/models/pricing: Pro and Standard are both $0.14/s, so only Pro.
SWAP_MODEL = {"id": "fal-ai/kling-video/o3/pro/video-to-video/edit",
              "label": "Kling O3 Pro Edit", "usd_per_second": 0.14}
SWAP_MAX_SECONDS = 15        # per call — longer videos are edited shot by shot
SWAP_MAX_TOTAL = 30
SWAP_MIN_SHORT_EDGE = 720
SWAP_MAX_ITEMS = 4


def options():
    return {
        "models": [{"key": k, "label": v["label"], "usd_per_second": v["usd_per_second"]}
                   for k, v in MODELS.items()],
        "default_model": DEFAULT_MODEL,
        "orientations": ORIENTATIONS,
        "min_seconds": MIN_SECONDS,
        "swap": {"label": SWAP_MODEL["label"], "usd_per_second": SWAP_MODEL["usd_per_second"],
                 "max_seconds": SWAP_MAX_TOTAL, "max_items": SWAP_MAX_ITEMS},
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


def prepare_video(raw, mode="motion"):
    """Re-encode the reference to H.264/AAC mp4 (≤1920 long edge), upload it to fal.

    mode "swap" applies the O3 Edit limits instead: up to 30 s (it is edited
    shot by shot, ≤15 s each), a short edge of at least 720 px (smaller clips are
    upscaled — fal rejects them otherwise), and returns the detected "shots".

    Returns {"url", "seconds", "width", "height", "thumb", "filename"}."""
    if not raw:
        raise MotionError("That video is empty.")
    if len(raw) > 500 * 1024 * 1024:
        raise MotionError("That video is over 500 MB. Trim it first.")

    os.makedirs(VIDEO_DIR, exist_ok=True)
    _cleanup_videos()
    max_s = SWAP_MAX_TOTAL if mode == "swap" else max(ORIENTATIONS.values())
    min_short = SWAP_MIN_SHORT_EDGE if mode == "swap" else 0
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        with open(src, "wb") as f:
            f.write(raw)
        _, info = video_assembler._run(["-i", src], timeout=60)
        m = re.search(r"Stream #\d+:\d+.*?: Video:.*?(\d{2,5})x(\d{2,5})", info)
        seconds = video_assembler._probe_duration(src)
        if not m or not seconds:
            raise MotionError("That file could not be read as a video. Use MP4 or MOV.")
        # Kling's own clips come out at 88 frames (2.93 s) — just under the 3 s
        # minimum. Anything that close gets its last frame held to reach it.
        pad = MIN_SECONDS + 0.1 - seconds if MIN_PAD_FROM <= seconds < MIN_SECONDS else 0
        if seconds < MIN_SECONDS and not pad:
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
        sw, sh = int(m.group(1)), int(m.group(2))
        if min_short and min(sw, sh) < min_short:
            # Upscale so the short edge reaches the minimum.
            scale = ("scale='if(gt(iw,ih),-2,%d)':'if(gt(iw,ih),%d,-2)'"
                     % (min_short, min_short))
        if pad:
            scale += ",tpad=stop_mode=clone:stop_duration=%.2f" % (pad + 0.1)
        args = ["-y", "-i", src, "-t", str(max_s if not pad else MIN_SECONDS + 0.1),
                "-vf", scale,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        args += ((["-af", "apad"] if pad else []) + ["-c:a", "aac", "-b:a", "128k"]
                 if has_audio else ["-an"])
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

    if mode == "swap":
        # The shots are uploaded one by one at generation time.
        shots = detect_shots(dest, seconds)
        return {"url": None, "seconds": seconds, "width": w, "height": h, "thumb": thumb,
                "filename": "%s/%s" % (VIDEO_SUBDIR, name), "has_audio": has_audio,
                "shots": shots, "billed_seconds": round(billed_seconds(shots), 2),
                "cost": estimate_swap_cost(shots)}
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


# ── Objects swap ───────────────────────────────────────────────────────────
#
# Kling O3 Edit loses track of what it is replacing at a hard cut: on a 4-shot
# ad (17.11.2026) it swapped the person in shots 1–2 and left the original man
# in shots 3–4. So the video is split at its cuts and every shot is edited on
# its own (in parallel), checked by Claude, retried once if the swap did not
# take, trimmed back and glued together under the original sound.

SHOT_SCENE_THRESHOLD = 0.3   # ffmpeg scene score; 0.2–0.4 gave identical cuts on real ads
SHOT_MIN = 0.4               # cuts closer than this are flashes, not shots
CHECK_MODEL = "claude-opus-5"
CHECK_FRAMES = (0.15, 0.5, 0.85)
SWAP_RETRIES = 1


def detect_shots(path, seconds):
    """[[start, end], ...] covering the whole video, split at hard cuts and
    so that no piece is longer than the model accepts."""
    _, err = video_assembler._run(
        ["-i", path, "-vf", "select=gt(scene\\,%s),showinfo" % SHOT_SCENE_THRESHOLD,
         "-an", "-f", "null", "-"], timeout=600)
    cuts = []
    for t in sorted(float(x) for x in re.findall(r"pts_time:([\d.]+)", err)):
        prev = cuts[-1] if cuts else 0.0
        if t - prev >= SHOT_MIN and seconds - t >= SHOT_MIN:
            cuts.append(round(t, 3))
    edges = [0.0] + cuts + [round(float(seconds), 3)]
    shots = []
    for a, b in zip(edges, edges[1:]):
        n = int((b - a) // (SWAP_MAX_SECONDS + 0.01)) + 1
        step = (b - a) / n
        for k in range(n):
            shots.append([round(a + k * step, 3), round(a + (k + 1) * step, 3)])
    return shots


def billed_seconds(shots):
    return sum(max(b - a, MIN_SECONDS) for a, b in shots)


def estimate_swap_cost(shots_or_seconds):
    if isinstance(shots_or_seconds, (list, tuple)):
        secs = billed_seconds(shots_or_seconds)
    else:
        secs = max(float(shots_or_seconds or 0), MIN_SECONDS)
    return round(secs * SWAP_MODEL["usd_per_second"], 2)


def clean_swaps(swaps):
    """[{what, url, extras}] → validated list, or MotionError."""
    out = []
    for i, sw in enumerate(swaps or []):
        what = re.sub(r"\s+", " ", str((sw or {}).get("what") or "")).strip().rstrip(".")
        # People type "Replace the man…"; the prompt already says "Replace".
        what = re.sub(r"^(please\s+)?(replace|swap|change)\s+", "", what, flags=re.I).strip()
        # "...on all scenes" / "in every frame": the shots are planned one by one anyway.
        what = re.sub(r"\s+(on|in|across|for|throughout)\s+(all|every|each|the whole)\s*(of\s+the\s+)?"
                      r"(scenes?|shots?|frames?|clips?|video)\b.*$", "", what, flags=re.I).strip() or what
        url = (sw or {}).get("url")
        if not what and not url:
            continue
        if not url:
            raise MotionError("Swap %d: add a photo of the new thing." % (i + 1))
        if not what:
            raise MotionError("Swap %d: say what in the video it replaces." % (i + 1))
        extras = [u for u in ((sw or {}).get("extras") or []) if u][:MAX_FACE_EXTRAS]
        out.append({"what": what[:200], "url": url, "extras": extras})
    if not out:
        raise MotionError("Add at least one swap: what to replace + a photo of the new thing.")
    if len(out) > SWAP_MAX_ITEMS:
        raise MotionError("At most %d swaps in one clip." % SWAP_MAX_ITEMS)
    return out


def build_swap_prompt(swaps, extra="", remove_text=False):
    # Only the swaps Claude saw in this shot reach here (_plan_shot): sending a
    # subject that is not in the shot made Kling paint it in (20.11.2026,
    # pillow ad — the unboxing came back as a copy of the woman's photo).
    # "in every frame" stays: the softer "wherever it appears" left the
    # people unchanged. Each element is tied to its own subject only, so
    # things that happen to be in its photo stay out.
    lines = ["Edit @Video1."]
    for i, sw in enumerate(swaps, 1):
        lines.append("Replace %s with @Element%d in every frame, matching the reference photos "
                     "exactly (face, hair, clothes, colour, shape); take only that subject from "
                     "the photos, not their background or other objects." % (sw.get("subject") or sw["what"], i))
    if remove_text:
        lines.append("Remove all on-screen text, captions, subtitles, stickers and emojis "
                     "overlaid on the video and show the clean scene behind them.")
        keep_tail = ""
    else:
        keep_tail = " and any on-screen text"
    lines.append("Keep everything else exactly as in @Video1: the action, motion, timing, camera, "
                 "framing, background, lighting, all other people and objects" + keep_tail + ". "
                 "Do not add any person or object that is not already in @Video1.")
    extra = (extra or "").strip()
    if extra:
        lines.append(extra)
    return " ".join(lines)


def build_swap_payload(video_url, swaps, extra="", remove_text=False):
    # Sound is put back from the original after the shots are joined, so the
    # model never has to carry it.
    payload = {"video_url": video_url,
               "prompt": build_swap_prompt(swaps, extra, remove_text),
               "keep_audio": False}
    if swaps:
        payload["elements"] = [{"frontal_image_url": sw["url"],
                                "reference_image_urls": sw["extras"] or [sw["url"]]}
                               for sw in swaps]
    return payload


def _source_path(video_file):
    root = os.path.realpath(video_assembler.OUTPUT_DIR)
    path = os.path.realpath(os.path.join(root, (video_file or "").replace("\\", "/")))
    if not video_file or not path.startswith(root + os.sep) or not os.path.isfile(path):
        raise MotionError("The uploaded video is no longer on the server — choose it again.")
    return path


def _cut_shot(src, a, b, dest):
    """One shot as its own mp4; shorter than the model minimum → last frame held."""
    dur = b - a
    # No setpts here: with ffmpeg 7.1, setpts before tpad silently drops the padding.
    vf = "null"
    if dur < MIN_SECONDS:
        vf = "tpad=stop_mode=clone:stop_duration=%.2f" % (MIN_SECONDS + 0.2 - dur)
    code, err = video_assembler._run(
        ["-y", "-ss", "%.3f" % a, "-t", "%.3f" % dur, "-i", src, "-an", "-vf", vf,
         "-t", "%.3f" % max(dur, MIN_SECONDS + 0.1),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", dest], timeout=600)
    if code != 0 or not os.path.exists(dest):
        raise MotionError("Could not cut shot %.1f–%.1fs: %s" % (a, b, err.strip()[-200:]))


def _frames_jpeg(path, dur, points):
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(points):
            f = os.path.join(tmp, "f%d.jpg" % i)
            video_assembler._run(["-y", "-ss", "%.2f" % (dur * p), "-i", path, "-frames:v", "1",
                                  "-vf", "scale=-2:720", f], timeout=60)
            if os.path.exists(f):
                with open(f, "rb") as fh:
                    out.append(fh.read())
    return out


def _img(raw):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(raw).decode()}}


def _claude_json(content, max_tokens=4000):
    import anthropic
    import json as _json
    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    resp = client.messages.create(model=CHECK_MODEL, max_tokens=max_tokens,
                                  messages=[{"role": "user", "content": content}])
    # Opus thinks first; a small max_tokens can be eaten by the thinking and
    # leave no answer at all (20.11.2026: 300 → empty 3 times out of 4).
    txt = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError("no JSON in the answer (%s)" % resp.stop_reason)
    return _json.loads(m.group(0))


_REF_CACHE = {}


def _ref_image(url):
    """The reference photo as JPEG bytes (downloaded once per run)."""
    if url not in _REF_CACHE:
        import requests as _rq
        raw = _rq.get(url, timeout=60).content
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        _REF_CACHE[url] = buf.getvalue()
    return _REF_CACHE[url]


def _plan_shot(src_shot, dur, swaps):
    """Which swaps are really in this ORIGINAL shot, and what exactly each one
    is here -> (swaps for this shot, each with a "subject" description, note).

    Sending an element for a subject that is not in the shot makes Kling
    invent it. The plan sees the reference photos: with only the words "the
    product" it took the plain white bed pillows of the "before" shot for the
    product (23.11.2026, abuela pillow ad). The concrete description also goes
    to Kling instead of the vague words: "the product" alone left a look-alike
    original unchanged. If the look itself fails, every swap is sent (old way)."""
    try:
        frames = _frames_jpeg(src_shot, dur, CHECK_FRAMES)
        if not frames:
            return list(swaps), ""
        content = []
        for i, sw in enumerate(swaps, 1):
            content += [{"type": "text", "text": "REFERENCE %d - the NEW thing that will replace \"%s\":"
                         % (i, sw["what"])}, _img(_ref_image(sw["url"]))]
        for i, fr in enumerate(frames, 1):
            content += [{"type": "text", "text": "FRAME %d of the original shot:" % i}, _img(fr)]
        items = "\n".join('%d. "%s" -> becomes REFERENCE %d' % (i, sw["what"], i)
                          for i, sw in enumerate(swaps, 1))
        content.append({"type": "text", "text":
            "A user wants these things in a video replaced by the reference photos:\n%s\n\n"
            "For each line, find in this shot the ONE thing the user means: the subject of the same "
            "kind as its reference (the same type of product, the same role of person). Use the "
            "reference to understand vague words like \"the product\" or \"the avatar\": a "
            "different kind of object that merely sits nearby (e.g. ordinary bed pillows when the "
            "reference is a backrest pillow with arms) does NOT count - then that line is not in "
            "this shot. A person who is not in the shot does not count, and hands alone do not count "
            "as a person.\n"
            "For each line that IS in the shot, describe that original subject as it looks here so "
            "a video editor can find it: kind, colour, material, where it is (max 20 words, no "
            "mention of the reference).\n"
            'Answer with JSON only: {"present": [{"n": <line number>, "subject": "<description>"}], '
            '"note": "<one short sentence>"}' % items})
        data = _claude_json(content)
        if not isinstance(data.get("present"), list):
            raise ValueError("no present list")
        out, seen = [], set()
        for it in data["present"]:
            if isinstance(it, dict):
                n, subj = it.get("n"), str(it.get("subject") or "").strip()
            else:
                n, subj = it, ""
            if not str(n).isdigit() or not 1 <= int(n) <= len(swaps) or int(n) in seen:
                continue
            seen.add(int(n))
            sw = dict(swaps[int(n) - 1])
            if subj:
                sw["subject"] = "%s (%s)" % (sw["what"], subj.rstrip(".")[:200])
            out.append((int(n), sw))
        return [sw for _, sw in sorted(out, key=lambda x: x[0])], str(data.get("note") or "")
    except Exception as e:
        return list(swaps), "(scene look skipped: %s)" % type(e).__name__


def _check_shot(path, dur, swaps, remove_text, src_shot=None):
    """Ask Claude whether the swap took in this shot and it is still the
    original shot (same action and framing, nothing invented). (ok, note).

    Never blocks the clip: if the check itself fails, the shot passes."""
    try:
        frames = _frames_jpeg(path, dur, CHECK_FRAMES)
        if not frames:
            return True, ""
        orig = _frames_jpeg(src_shot, dur, CHECK_FRAMES) if src_shot else []
        content = []
        for i, sw in enumerate(swaps, 1):
            content += [{"type": "text", "text": "REFERENCE %d — what \"%s\" must become:"
                         % (i, sw.get("subject") or sw["what"])}, _img(_ref_image(sw["url"]))]
        for i, fr in enumerate(orig, 1):
            content += [{"type": "text", "text": "ORIGINAL FRAME %d (before the edit):" % i}, _img(fr)]
        for i, fr in enumerate(frames, 1):
            content += [{"type": "text", "text": "EDITED FRAME %d (same moment):" % i}, _img(fr)]
        if swaps:
            asks = "; ".join('"%s" → REFERENCE %d' % (sw.get("subject") or sw["what"], i)
                             for i, sw in enumerate(swaps, 1))
            rule = ("This shot was edited to replace: %s.\n"
                    "Fail it if, in any frame where that subject is visible, it is still the original "
                    "one instead of the reference (same person / same product). Compare with the "
                    "ORIGINAL frames: a product that kept the original's colour, material or details "
                    "instead of the reference's is NOT swapped, even if the shape is similar.\n" % asks)
        else:
            rule = "This shot was edited only to remove overlaid text.\n"
        same = ("Also fail it if the edited shot is not the same scene as the original: a different "
                "action, a different camera angle or framing, or a person or object added that was "
                "not in the original frames, or any OTHER object changed into something else (only "
                "the subjects listed above may change; e.g. plain bed pillows turned into the "
                "product). Changes caused by the swap itself are fine.\n"
                if orig else "")
        text_rule = ("Also fail it if any overlaid caption, subtitle or sticker text is still "
                     "visible (text that is part of the scene itself, like a sign or a logo on "
                     "clothing, is fine).\n" if remove_text else "")
        content.append({"type": "text", "text": rule + same + text_rule +
            'Answer with JSON only: {"ok": true|false, "score": <0-10, how well the edit did '
            'all of the above>, "note": "<one short sentence>"}'})
        data = _claude_json(content)
        try:
            score = float(data.get("score"))
        except (TypeError, ValueError):
            score = 5.0
        return bool(data.get("ok", True)), str(data.get("note") or ""), score
    except Exception as e:
        return True, "(check skipped: %s)" % type(e).__name__, 5.0


def _edit_shot(idx, n, src_shot, dur, swaps, extra, remove_text, tmp, say):
    """Look, upload, edit, check, maybe retry. Returns (local_path, cost, warning)."""
    tag = "Shot %d/%d" % (idx + 1, n) if n > 1 else "Clip"
    mine, pnote = _plan_shot(src_shot, dur, swaps)
    say("%s: 👀 %s%s" % (tag, "; ".join(sw.get("subject") or sw["what"] for sw in mine)
                         if mine else "nothing to replace",
                         (" — " + pnote) if pnote else ""))
    if not mine and not remove_text:
        say("%s: kept as in the original (free)" % tag)
        return src_shot, 0.0, ""
    with open(src_shot, "rb") as fh:
        url = _upload(fh.read(), os.path.basename(src_shot), "video/mp4")
    payload = build_swap_payload(url, mine, extra, remove_text)
    billed = max(dur, MIN_SECONDS) * SWAP_MODEL["usd_per_second"]
    cost, warning, out_path = 0.0, "", None
    best = None   # (score, path, warning) — a retry can come back worse than the first try
    for attempt in range(SWAP_RETRIES + 1):
        res = fal_client.run(SWAP_MODEL["id"], payload, timeout=2400,
                             on_status=lambda t: say("%s: %s" % (tag, t)))
        cost += billed
        video = res.get("video") or {}
        vurl = video.get("url") if isinstance(video, dict) else None
        if not vurl:
            raise MotionError("%s: the model returned no video." % tag)
        out_path = os.path.join(tmp, "out_%02d_%d.mp4" % (idx, attempt))
        with open(out_path, "wb") as fh:
            fh.write(fal_client.download(vurl))
        ok, note, score = _check_shot(out_path, dur, mine, remove_text, src_shot)
        if ok:
            say("%s: ✅ swap checked%s" % (tag, (" — " + note) if note.startswith("(") else ""))
            return out_path, cost, ""
        warning = "%s: ⚠️ %s" % (tag, note or "the swap did not take")
        if best is None or score > best[0]:
            best = (score, out_path, warning)
        if attempt < SWAP_RETRIES:
            say(warning + " — trying this shot again…")
    _, out_path, warning = best
    say(warning + " (kept the better try)")
    return out_path, cost, warning


def _join(parts, durs, src, keep_sound, dest):
    """Trim every edited shot to its real length, scale to one size/fps, concat,
    and put the original sound back."""
    _, info = video_assembler._run(["-i", parts[0]], timeout=60)
    m = re.search(r"Stream #\d+:\d+.*?: Video:.*?(\d{2,5})x(\d{2,5})", info)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (1080, 1920)
    args = ["-y"]
    for p in parts:
        args += ["-i", p]
    if keep_sound:
        args += ["-i", src]
    chains = []
    for i, d in enumerate(durs):
        chains.append("[%d:v]trim=duration=%.3f,setpts=PTS-STARTPTS,scale=%d:%d:force_original_aspect_ratio=decrease,"
                      "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1[v%d]" % (i, d, w, h, w, h, i))
    chains.append("".join("[v%d]" % i for i in range(len(parts))) +
                  "concat=n=%d:v=1:a=0[vout]" % len(parts))
    args += ["-filter_complex", ";".join(chains), "-map", "[vout]"]
    if keep_sound:
        args += ["-map", "%d:a?" % len(parts), "-c:a", "aac", "-b:a", "160k", "-shortest"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", dest]
    code, err = video_assembler._run(args, timeout=900)
    if code != 0 or not os.path.exists(dest):
        raise MotionError("Could not join the shots: %s" % err.strip()[-300:])


def swap_stream(video_file, seconds, swaps, shots=None, extra="", keep_sound=True,
                remove_text=False, video_url=None):
    """Make one objects-swap clip. Yields {"type": "status"|"done"} events."""
    import queue
    from concurrent.futures import ThreadPoolExecutor

    try:
        swaps = clean_swaps(swaps)
        src = _source_path(video_file)
        seconds = float(seconds or 0) or (video_assembler._probe_duration(src) or 0)
        if seconds > SWAP_MAX_TOTAL + 0.5:
            raise MotionError("Objects swap allows at most %ds of video." % SWAP_MAX_TOTAL)
        shots = shots or detect_shots(src, seconds)
    except MotionError as e:
        yield {"type": "done", "error": str(e)}
        return

    n = len(shots)
    est = estimate_swap_cost(shots)
    yield {"type": "status", "text": "🔁 %s · %.1fs · %d shot%s · %d swap%s%s · ≈$%.2f" % (
        SWAP_MODEL["label"], seconds, n, "" if n == 1 else "s", len(swaps),
        "" if len(swaps) == 1 else "s", " · text removed" if remove_text else "", est)}
    if n > 1:
        yield {"type": "status", "text": "✂️ Cuts at " + ", ".join("%.1fs" % a for a, _ in shots[1:])
               + " — each shot is edited on its own so the swap holds after every cut"}
    yield {"type": "status", "text": "📝 " + build_swap_prompt(swaps, extra, remove_text)}
    yield {"type": "status", "text": "👀 Claude first looks at every shot and sends Kling only "
                                     "the swaps that are really in it"}
    yield {"type": "status", "text": "⏳ Kling usually takes 5–10 minutes (shots run at the same time)…"}

    msgs = queue.Queue()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cut = []
            for i, (a, b) in enumerate(shots):
                p = os.path.join(tmp, "shot_%02d.mp4" % i)
                _cut_shot(src, a, b, p)
                cut.append(p)
            with ThreadPoolExecutor(max_workers=min(n, 6)) as pool:
                futs = [pool.submit(_edit_shot, i, n, cut[i], b - a, swaps, extra, remove_text,
                                    tmp, msgs.put) for i, (a, b) in enumerate(shots)]
                while not all(f.done() for f in futs):
                    try:
                        yield {"type": "status", "text": "   " + msgs.get(timeout=2)}
                    except queue.Empty:
                        pass
                while not msgs.empty():
                    yield {"type": "status", "text": "   " + msgs.get()}
                results = [f.result() for f in futs]

            cost = round(sum(r[1] for r in results), 2)
            warnings = [r[2] for r in results if r[2]]
            yield {"type": "status", "text": "🧩 Joining the shots…" if n > 1 else "🎞️ Finishing…"}
            name = "clip_swap_%s_%d.mp4" % (clip_studio._safe_stem(swaps[0]["what"]), int(time.time()))
            _join([r[0] for r in results], [b - a for a, b in shots], src, keep_sound,
                  os.path.join(video_assembler.OUTPUT_DIR, name))
        yield {"type": "status", "text": "✅ Done" + (" — with warnings above" if warnings else "")}
        yield {"type": "done", "filename": name, "cost": cost, "seconds": seconds,
               "mode": "swap", "model_label": SWAP_MODEL["label"], "shots": n,
               "swaps": [sw["what"] for sw in swaps], "keep_sound": bool(keep_sound),
               "remove_text": bool(remove_text), "warnings": warnings,
               "prompt": build_swap_prompt(swaps, extra, remove_text)}
    except (fal_client.FalError, MotionError) as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": "%s: %s" % (type(e).__name__, e)}
