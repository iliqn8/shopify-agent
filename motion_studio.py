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


def _slots(swaps):
    """Kling O3 Edit: people/objects go in `elements` (@ElementN), places in
    `image_urls` (@ImageN); at most 4 in total. A place sends its main photo
    plus its extra angles while they fit — a photo taken from the shot's own
    camera angle is what makes Kling lay the room out right (24.11.2026: from
    a side view of the room it put a second bed behind her instead of the
    headboard right behind her). Returns (elements, image_urls, tags)."""
    el = [sw for sw in swaps if sw.get("kind") != "place"]
    places = [sw for sw in swaps if sw.get("kind") == "place"]
    room = 4 - len(el)
    urls, tags = [], {}
    for i, sw in enumerate(el, 1):
        tags[id(sw)] = "@Element%d" % i
    for sw in places:                      # every place gets its main photo first
        if len(urls) < room:
            urls.append(sw["url"])
            tags[id(sw)] = ["@Image%d" % len(urls)]
    for sw in places:                      # then the extra angles, while they fit
        for u in sw.get("extras") or []:
            if len(urls) < room and id(sw) in tags:
                urls.append(u)
                tags[id(sw)].append("@Image%d" % len(urls))
    for sw in places:
        t = tags.get(id(sw)) or ["@Image1"]
        tags[id(sw)] = t[0] if len(t) == 1 else ", ".join(t[:-1]) + " and " + t[-1]
    return el, urls, tags


PROMPT_MAX = 2400   # Kling O3 Edit rejects prompts over 2500 characters (after the job ran)


def _words(text, n):
    w = (text or "").split()
    return " ".join(w[:n]).rstrip(".,;") if w else ""


def build_swap_prompt(swaps, extra="", remove_text=False):
    # Rules below each come from a clip that went wrong:
    # - only the swaps Claude saw in this shot reach here (_plan_shot); a
    #   subject that is not in the shot gets painted in (20.11, pillow unboxing);
    # - "in every frame" stays: "wherever it appears" left people unchanged;
    # - people are named by place/action, the outfit is asked for outright —
    #   with "woman in a cream cardigan" Kling kept the cardigan (23.11, abuela);
    # - a place is an @Image, not an element: as an element the new room came
    #   and went between shots (24.11, pillow 14);
    # - a place lists what the new one has and what of the old must go — with
    #   only "move the scene into @Image1" a close shot kept its room (24.11);
    # - the furniture people are on keeps its role ("support"): "repaint the
    #   bedding too" turned the bed she sat on into a floor rug (24.11, live);
    # - what is inside/on an object follows the video, not the photo — the ad
    #   photo's phones ended up in the pockets (24.11, pillow 14);
    # - never an already edited frame as a reference: Kling copies it whole.
    # Kling refuses > 2500 characters, so descriptions shrink until it fits.
    el, im, tags = _slots(swaps)
    extra = (extra or "").strip()
    for budget in (45, 30, 20, 12, 6, 0):
        lines = ["Edit @Video1."]
        for sw in swaps:
            who, tag = sw.get("subject") or sw["what"], tags[id(sw)]
            if budget < 20 and "(" in who:
                who = sw["what"] + (" (" + _words(who[who.find("(") + 1:who.rfind(")")], budget) + ")"
                                    if budget else "")
            kind = sw.get("kind") or ("person" if sw.get("person") else "object")
            if kind == "person":
                lines.append("Replace %s with the person %s in every frame: face, hair, body and the "
                             "complete outfit (every garment, colour, pattern, accessory) exactly as "
                             "in the photos; none of the original clothes may remain." % (who, tag))
            elif kind == "place":
                old = sw.get("subject") or ""
                old = old[old.find("(") + 1:old.rfind(")")] if "(" in old else ""
                look, support = _words(sw.get("look"), budget), _words(sw.get("support"), budget)
                old = _words(old, budget)
                views = (" (the same place from different angles — follow the view closest to "
                         "this shot's camera angle and copy its layout)") if " and " in tag else ""
                lines.append("Restyle the setting in every frame into the place in %s%s%s, as if "
                             "filmed there: each piece keeps its role and position (walls, headboard, "
                             "bedding, furniture, decor), even out of focus; nothing of the original "
                             "look%s may remain." % (tag, views, (" (" + look + ")") if look else "",
                                                    (" (" + old + ")") if old else ""))
                if support:
                    lines.append("In @Video1 %s — keep that: a bed stays a bed, the surface under "
                                 "them becomes its bedding, never a floor or rug, and no second bed "
                                 "appears behind them." % support)
            else:
                lines.append("Replace %s with %s in every frame, exactly matching the photos (colour, "
                             "material, shape, details); what is inside or on it (e.g. items in its "
                             "pockets) stays as in @Video1, nothing added from the photos."
                             % (who, tag))
        lines.append("Take only the named subject from each photo, not its background or other "
                     "objects.")
        if remove_text:
            lines.append("Remove all overlaid text, captions, stickers and emojis.")
        lines.append("Keep the action, motion, timing, camera, framing, lighting%s and all other "
                     "people and objects exactly as in @Video1; replaced subjects keep the original "
                     "pose and movement; add no new person or object."
                     % (("" if im else ", background") + ("" if remove_text else ", on-screen text")))
        if extra:
            lines.append(extra)
        prompt = " ".join(lines)
        if len(prompt) <= PROMPT_MAX:
            return prompt
    return prompt[:PROMPT_MAX]


def build_swap_payload(video_url, swaps, extra="", remove_text=False):
    # Sound is put back from the original after the shots are joined, so the
    # model never has to carry it.
    el, images, _ = _slots(swaps)
    payload = {"video_url": video_url,
               "prompt": build_swap_prompt(swaps, extra, remove_text),
               "keep_audio": False}
    if el:
        payload["elements"] = [{"frontal_image_url": sw["url"],
                                "reference_image_urls": (sw.get("extras") or [sw["url"]])[:3]}
                               for sw in el]
    if images:
        payload["image_urls"] = images
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


CLEAN_MODEL = "fal-ai/nano-banana-pro/edit"
CLEAN_USD = 0.15
_CLEAN_CACHE = {}


def _classify(urls_whats):
    """Claude: kind (person/object/place) + which thing is meant, per photo.
    [(url, what)] → [{"kind", "keep"}]; falls back to object / the user's words."""
    content = []
    for i, (url, what) in enumerate(urls_whats, 1):
        content += [{"type": "text", "text": 'PHOTO %d — the user wrote: "%s"' % (i, what)},
                    _img(_ref_image(url))]
    content.append({"type": "text", "text":
        "Each photo shows the NEW thing that will replace something in a video.\n"
        "For each photo answer: kind = \"person\", \"object\" (a product or prop) or \"place\" "
        "(a room, setting or background); keep = which single thing in the photo is meant "
        "(max 15 words, e.g. \"the light-blue ribbed backrest pillow with arms\"); "
        "look = ONLY for a place: the 6-10 most visible things in it that a video editor "
        "must paint in, with colours (max 45 words, e.g. \"large US flag above a beige "
        "tufted headboard, pine garland with warm lights and a red bow, red plaid pillows, "
        "red knit blanket, lit table lamps, decorated Christmas tree by a snowy window\"); "
        "empty for others.\n"
        'JSON only: {"photos": [{"n": 1, "kind": "...", "keep": "...", "look": "..."}]}'})
    out = [{"kind": "object", "keep": what} for _, what in urls_whats]
    try:
        data = _claude_json(content)
        for x in data.get("photos") or []:
            n = int(x.get("n")) if str(x.get("n")).isdigit() else 0
            if 1 <= n <= len(out):
                if x.get("kind") in ("person", "object", "place"):
                    out[n - 1]["kind"] = x["kind"]
                if x.get("keep"):
                    out[n - 1]["keep"] = str(x["keep"])[:150]
                if x.get("look") and out[n - 1]["kind"] == "place":
                    out[n - 1]["look"] = str(x["look"])[:350]
    except Exception:
        pass
    return out


def _clean_one(url, kind, keep):
    if kind == "person":
        prompt = ("Show ONLY %s from this image: exactly the same person — same face, skin, hair, "
                  "body and the complete outfit with every colour, pattern, print and logo on the "
                  "clothes — full body, standing, alone on a plain pure white background, hands "
                  "empty. Remove the room, overlaid text and every other object or person." % keep)
        ar = "3:4"
    else:
        # The product's OWN logos, labels and prints stay — the user sells
        # branded products. Only what is not part of the product goes.
        prompt = ("Product photo: show ONLY %s from this image, complete and exactly as it is — "
                  "same colour, material, texture, stitching, shape, proportions and details, "
                  "INCLUDING any logo, label, print or text that is printed on or attached to the "
                  "product itself. Put it alone and fully in frame on a plain pure white "
                  "background. Remove only what is not part of the product: overlaid ad text, "
                  "badges and graphics, the room or background, other products or colour "
                  "versions, and items placed in, on or held by it (phones, remotes, bottles, "
                  "cups, books) — pockets and holders empty." % keep)
        ar = "1:1"
    return fal_client.edit_image([url], prompt, aspect_ratio=ar, model_id=CLEAN_MODEL)


def clean_photo(url, what=""):
    """🧼 button: a clean cut-out of the uploaded photo (only on request — a
    product's own logo must be copyable as is). Same photo → cached, free.
    Returns {"url", "thumb", "width", "height", "cleaned", "cost"}."""
    if not url or not str(url).startswith("http"):
        raise MotionError("Upload the photo first.")
    if url in _CLEAN_CACHE:
        return dict(_CLEAN_CACHE[url], cost=0.0)
    info = _classify([(url, what or "the main subject")])[0]
    if info["kind"] == "place":
        raise MotionError("This looks like a room/setting — those are used as they are, no cleaning needed.")
    try:
        new = _clean_one(url, info["kind"], info["keep"])
        raw = fal_client.download(new)
    except fal_client.FalError as e:
        raise MotionError("Cleaning failed: %s" % str(e)[:200])
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    t = im.copy()
    t.thumbnail((240, 240))
    tb = io.BytesIO()
    t.save(tb, "JPEG", quality=80)
    res = {"url": new, "width": im.width, "height": im.height, "cleaned": True, "kind": info["kind"],
           "thumb": "data:image/jpeg;base64," + base64.b64encode(tb.getvalue()).decode()}
    _CLEAN_CACHE[url] = res
    return dict(res, cost=CLEAN_USD)


def _prepare_refs(swaps, say):
    """Sort every reference into person / object / place (one Claude call) —
    places go to Kling as @Image, people get the outfit rule. Photos are used
    exactly as uploaded; cleaning is the user's 🧼 choice per photo, because
    branded products must keep their logo (24.11.2026). Returns (swaps, 0)."""
    out = [dict(sw) for sw in swaps]
    infos = _classify([(sw["url"], sw["what"]) for sw in out])
    for i, (sw, info) in enumerate(zip(out, infos), 1):
        sw["kind"] = info["kind"]
        sw["person"] = info["kind"] == "person"
        if info.get("look"):
            sw["look"] = info["look"]
        say("📷 Photo %d: %s — %s" % (i, info["kind"], info["keep"]))
    return out, 0.0


def _place_views(sw):
    """A place's photos: the main one, then its extra angles."""
    if sw.get("kind") != "place":
        return [sw["url"]]
    return [sw["url"]] + [u for u in (sw.get("extras") or []) if u]


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
        content, multi = [], False
        for i, sw in enumerate(swaps, 1):
            views = _place_views(sw)
            if len(views) > 1:
                multi = True
                for k, u in enumerate(views, 1):
                    content += [{"type": "text", "text": "REFERENCE %d, VIEW %d - the NEW place that "
                                 "will replace \"%s\", seen from one angle:" % (i, k, sw["what"])},
                                _img(_ref_image(u))]
            else:
                content += [{"type": "text", "text": "REFERENCE %d - the NEW thing that will replace \"%s\":"
                             % (i, sw["what"])}, _img(_ref_image(sw["url"]))]
        view_rule = ("A place shown in several VIEWS: also give \"view\" - the number of the view whose "
                     "camera angle and distance are closest to this shot (e.g. a close shot of someone "
                     "in front of a headboard -> the close view of that headboard, not the wide room "
                     "view).\n" if multi else "")
        for i, fr in enumerate(frames, 1):
            content += [{"type": "text", "text": "FRAME %d of the original shot:" % i}, _img(fr)]
        items = "\n".join('%d. "%s" (%s) -> becomes REFERENCE %d'
                          % (i, sw["what"], sw.get("kind") or "?", i)
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
            "a video editor can find it (max 20 words, no mention of the reference). For an object: "
            "kind, colour, material, where it is. For a person: only who and where (e.g. \"the "
            "elderly woman sitting on the bed\") - do NOT describe their clothes, hair or face, "
            "because those are being replaced. For a place: it is present whenever any of the "
            "setting is visible; describe the visible setting (walls, bed, furniture, decor), and "
            "also give \"support\": what the people in the shot sit, lie or stand on and what is "
            "right behind them (max 25 words, e.g. \"she sits ON the bed, on its floral bedspread; "
            "the bed's carved headboard is right behind her\").\n%s"
            'Answer with JSON only: {"present": [{"n": <line number>, "person": true|false, '
            '"subject": "<description>", "support": "<only for a place>", "view": <only for a '
            'place shown in views>}], "note": "<one short sentence>"}' % (items, view_rule)})
        data = _claude_json(content)
        if not isinstance(data.get("present"), list):
            raise ValueError("no present list")
        out, seen = [], set()
        for it in data["present"]:
            person, support = False, ""
            if isinstance(it, dict):
                n, subj = it.get("n"), str(it.get("subject") or "").strip()
                support = str(it.get("support") or "").strip()
                person = it.get("person") is True or str(it.get("person")).lower() == "true"
            else:
                n, subj = it, ""
            if not str(n).isdigit() or not 1 <= int(n) <= len(swaps) or int(n) in seen:
                continue
            seen.add(int(n))
            sw = dict(swaps[int(n) - 1])
            if sw.get("kind"):
                person = sw["kind"] == "person"
            sw["person"] = person
            if subj:
                sw["subject"] = "%s (%s)" % (sw["what"], subj.rstrip(".")[:200])
            if support and sw.get("kind") == "place":
                sw["support"] = support.rstrip(".")[:220]
            views = _place_views(sw)
            if len(views) > 1:
                v = str(it.get("view") or "1") if isinstance(it, dict) else "1"
                v = int(v) if v.isdigit() and 1 <= int(v) <= len(views) else 1
                # Kling lays the room out like the FIRST photo it gets, so only
                # the closest view goes (24.11.2026, pillow 14 shot 3: wide photo
                # + close view -> second bed behind her; close view alone ->
                # headboard right behind her, as in the original).
                if v > 1:
                    sw.pop("look", None)   # the wide photo's list would add its tree/window
                sw["url"], sw["extras"] = views[v - 1], []
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
            return True, "", 5.0
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
                    "instead of the reference's is NOT swapped, even if the shape is similar. A person "
                    "who got the new face and hair but still wears the ORIGINAL clothes is NOT "
                    "swapped: the outfit must be the reference's too. For a place/setting: if walls, "
                    "furniture, bedding or decor of the ORIGINAL setting are still visible instead "
                    "of the reference's, it is NOT swapped.\n"
                    "Also fail it if items appear in, on or around a replaced object that were not "
                    "in the original frames (e.g. phones or remotes in pockets that were empty or "
                    "held other things).\n" % asks)
        else:
            rule = "This shot was edited only to remove overlaid text.\n"
        same = ("Also fail it if the edited shot is not the same scene as the original: a different "
                "action, a different camera angle or framing, or a person or object added that was "
                "not in the original frames, or any OTHER object changed into something else (only "
                "the subjects listed above may change; e.g. plain bed pillows turned into the "
                "product), or a person who now sits, lies or stands on something else than in the "
                "original (e.g. was on a bed, now on the floor or a rug, with a bed behind her). "
                "Changes caused by the swap itself are fine.\n"
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


FAL_RESUBMITS = 2


def _run_kling(payload, tag, say):
    """fal_client.run, submitted again when the job dies on fal's side."""
    for k in range(FAL_RESUBMITS + 1):
        try:
            return fal_client.run(SWAP_MODEL["id"], payload, timeout=2400,
                                  on_status=lambda t: say("%s: %s" % (tag, t)))
        except fal_client.FalTransient as e:
            if k == FAL_RESUBMITS:
                raise
            say("%s: ⚠️ fal/Kling hiccup (%s) — sending this shot again…" % (tag, str(e)[:80]))
            time.sleep(10)


def _edit_shot_safe(idx, n, src_shot, dur, mine, extra, remove_text, tmp, say):
    """_edit_shot, but a shot that cannot be made keeps the original footage
    instead of throwing away every finished (and paid) shot."""
    try:
        return _edit_shot(idx, n, src_shot, dur, mine, extra, remove_text, tmp, say)
    except (fal_client.FalError, MotionError) as e:
        tag = "Shot %d/%d" % (idx + 1, n) if n > 1 else "Clip"
        if n == 1:
            raise
        warning = "%s: ❌ could not be edited (%s) — the ORIGINAL footage is kept there" % (tag, str(e)[:120])
        say(warning)
        return src_shot, getattr(e, "_cost", 0.0), warning


def _edit_shot(idx, n, src_shot, dur, mine, extra, remove_text, tmp, say):
    """Upload, edit, check, maybe retry. `mine` = this shot's plan.
    Returns (local_path, cost, warning)."""
    tag = "Shot %d/%d" % (idx + 1, n) if n > 1 else "Clip"
    if not mine and not remove_text:
        say("%s: kept as in the original (free)" % tag)
        return src_shot, 0.0, ""
    with open(src_shot, "rb") as fh:
        url = _upload(fh.read(), os.path.basename(src_shot), "video/mp4")
    cost = 0.0
    payload = build_swap_payload(url, mine, extra, remove_text)
    billed = max(dur, MIN_SECONDS) * SWAP_MODEL["usd_per_second"]
    warning, out_path = "", None
    best = None   # (score, path, warning) — a retry can come back worse than the first try
    for attempt in range(SWAP_RETRIES + 1):
        try:
            res = _run_kling(payload, tag, say)
        except (fal_client.FalError, MotionError) as e:
            if best is not None:        # the first try is done and paid — use it
                say("%s: retry failed (%s) — keeping the first try" % (tag, str(e)[:80]))
                break
            e._cost = cost
            raise
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
    yield {"type": "status", "text": "👀 Then Claude looks at every shot and sends Kling only "
                                     "the swaps that are really in it"}
    yield {"type": "status", "text": "⏳ Kling usually takes 5–10 minutes (shots run at the same time)…"}

    msgs = queue.Queue()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with ThreadPoolExecutor(max_workers=1) as one:
                job = one.submit(_swap_pipeline, src, shots, swaps, extra, remove_text, tmp, msgs.put)
                while not job.done():
                    try:
                        yield {"type": "status", "text": "   " + msgs.get(timeout=2)}
                    except queue.Empty:
                        pass
                while not msgs.empty():
                    yield {"type": "status", "text": "   " + msgs.get()}
                results, prep_cost, prompt = job.result()

            cost = round(sum(r[1] for r in results) + prep_cost, 2)
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
               "prompt": prompt}
    except (fal_client.FalError, MotionError) as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": "%s: %s" % (type(e).__name__, e)}


def _swap_pipeline(src, shots, swaps, extra, remove_text, tmp, say):
    """Sort photos → plan every shot → edit all shots at the same time.
    Returns (results, prep_cost, prompt)."""
    from concurrent.futures import ThreadPoolExecutor
    n = len(shots)
    swaps, prep_cost = _prepare_refs(swaps, say)

    cut = []
    for i, (a, b) in enumerate(shots):
        p = os.path.join(tmp, "shot_%02d.mp4" % i)
        _cut_shot(src, a, b, p)
        cut.append(p)

    with ThreadPoolExecutor(max_workers=min(n, 6)) as pool:
        plans = list(pool.map(lambda i: _plan_shot(cut[i], shots[i][1] - shots[i][0], swaps), range(n)))
    for i, (mine, note) in enumerate(plans):
        tag = "Shot %d/%d" % (i + 1, n) if n > 1 else "Clip"
        say("%s: 👀 %s%s" % (tag, "; ".join(sw.get("subject") or sw["what"] for sw in mine)
                             if mine else "nothing to replace", (" — " + note) if note else ""))

    # NO continuity frame from an already edited shot: Kling O3 Edit copies
    # such an @Image whole — shots 1–2 came back as copies of shot 3 (pose,
    # props, framing), 24.11.2026 pillow 14. Every shot is edited only
    # against its own original footage.
    results = [None] * n
    with ThreadPoolExecutor(max_workers=max(1, min(n, 6))) as pool:
        futs = {i: pool.submit(_edit_shot_safe, i, n, cut[i], shots[i][1] - shots[i][0],
                               plans[i][0], extra, remove_text, tmp, say)
                for i in range(n)}
        for i, f in futs.items():
            results[i] = f.result()
    return results, prep_cost, build_swap_prompt(swaps, extra, remove_text)

