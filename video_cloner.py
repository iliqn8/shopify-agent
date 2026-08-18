"""Video Cloner — analyse a reference video, then rebuild the same format for our product.

Two phases, both exposed as generators so the UI can stream progress
(same job/poll pattern as section_builder):

  analyze_stream()  reference video -> "recipe" JSON (scenes, timing, hook, shots)
  generate_stream() recipe -> generated clips (fal.ai) -> assembled MP4

The recipe is deliberately a plain JSON document the user can read and edit in
the UI before spending any credits — generation is the expensive half.
"""

import os
import re
import json
import math
import base64
import shutil
import tempfile
import subprocess

import cv2
import anthropic

import fal_client
import avatar_registry
import video_assembler

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

MODEL = "claude-opus-4-8"

# Kling Pro sits at a fifth of Seedance's price without a matching drop in
# realism, so it is the default; Standard is cheaper still but visibly weaker.
DEFAULT_VIDEO_MODEL = "kling-2.6-pro"

# Reference frames sent to the vision model. More frames = better read on pacing
# and shot changes, at the cost of tokens.
ANALYSIS_FRAMES = 16


# ── Probing / frame extraction ─────────────────────────────────────────────

def probe_video(video_bytes):
    """Duration, fps, dimensions and aspect ratio of raw video bytes.

    Uses OpenCV rather than ffprobe — ffprobe is not bundled with the
    imageio-ffmpeg binary we rely on for assembly.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = round(total / fps, 2) if fps > 0 and total > 0 else 0.0
        if w and h:
            ratio = w / h
            aspect = "9:16" if ratio < 0.85 else ("1:1" if ratio < 1.2 else "16:9")
        else:
            aspect = "9:16"
        return {"duration": duration, "fps": round(fps, 2), "width": w, "height": h, "aspect_ratio": aspect}
    except Exception:
        return {"duration": 0.0, "fps": 30.0, "width": 0, "height": 0, "aspect_ratio": "9:16"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_frames(video_bytes, max_frames=ANALYSIS_FRAMES):
    """Evenly-spaced JPEG frames with their timestamps. Never raises."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total <= 0:
            cap.release()
            return []

        count = min(max_frames, total)
        indices = [int(i * (total - 1) / max(count - 1, 1)) for i in range(count)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            # Downscale wide side to 768px — plenty for layout/pacing reading.
            h, w = frame.shape[:2]
            scale = 768 / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok2:
                frames.append({
                    "b64": base64.b64encode(buf.tobytes()).decode(),
                    "media_type": "image/jpeg",
                    "t": round(idx / fps, 2) if fps > 0 else 0.0,
                })
        cap.release()
        return frames
    except Exception:
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _media_type(raw):
    """The real image type of a downloaded byte string.

    Claimed types are not interchangeable: the API rejects a PNG declared as
    JPEG outright. `_verify_swap` used to hardcode image/jpeg for every URL it
    fetched, so one PNG product photo turned the whole check into a 400, which
    the caller swallowed as "verification unavailable" and treated as a pass —
    silently disabling the swap check for every project with a PNG upload.
    """
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"GIF":
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _layout_similarity(url_a, url_b):
    """How much two images share a layout, 0..1. None if either can't be read.

    Compares coarse greyscale structure — where the big masses of light and
    dark sit — deliberately ignoring colour, so that recolouring a product does
    not read as recomposing the shot.

    It used to correlate a 96x96 Laplacian, and that was wrong in a way that
    cost real money on project 21. A Laplacian at that scale is dominated by
    fine texture — fur, ripples, the weave of a mat — and every one of these
    edit models repaints that texture. Two frames of the same shot then
    correlate noise against noise and score near zero, while the metric only
    scored well when the model had returned an almost untouched file. Measured
    on that project's shot 4: the one rung that produced exactly the right
    frame — same pose, same background, vest correctly replaced — scored 0.000,
    lower than a rung that returned a two-up collage of invented scenes (0.067).
    All four rungs were rejected structurally, before the vision check ever saw
    them, and the shot shipped with the reference's own red vest.

    Coarse structure survives repainting: on the same set the correct swaps
    score 0.71-1.00 and the collage, the redrawn shots and frames from other
    shots of the same ad score 0.00-0.51.
    """
    try:
        import numpy as np
        import requests as _rq

        mats = []
        for url in (url_a, url_b):
            if url.startswith("data:"):
                raw = base64.b64decode(url.split(",", 1)[1])
            else:
                raw = _rq.get(url, timeout=120).content
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                return None
            arr = cv2.resize(arr, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            # A light blur on top of the downscale: it costs nothing and takes
            # the edge off a model that returns the frame slightly re-cropped.
            arr = cv2.GaussianBlur(arr, (0, 0), 1.0)
            arr -= arr.mean()
            norm = float(np.linalg.norm(arr))
            if norm < 1e-6:
                return None
            mats.append(arr / norm)

        return float(np.clip((mats[0] * mats[1]).sum(), 0.0, 1.0))
    except Exception:
        return None


# What can optionally be replaced inside an otherwise faithful recreation.
# Each is independent and opt-in: with no photo uploaded for a slot, that part
# of the frame is left exactly as the reference had it.
#
# `layout_min` is the coarse-structure floor for accepting an edit. It has to
# differ per slot: replacing a product barely moves the frame's masses, but
# replacing the environment redraws most of them by design, so the structural
# check is useless there and the vision check carries it alone.
#
# These are floors for a CHEAP FIRST FILTER, not verdicts. Its job is to throw
# out the obvious disasters — a collage, a wholly different scene — without
# paying for a vision call; anything arguable belongs to `_verify_swap`, which
# is far better at it. Set too high, it rejects good frames before the good
# judge ever sees them, which is exactly what shipped shot 4 of project 21 with
# the wrong vest. Calibrated on that project's frames: correct swaps score
# 0.71-1.00, collages and different shots 0.00-0.51.
SWAP_SLOTS = {
    "product": {
        "noun": "product",
        "label": "product",
        "layout_min": 0.45,
        "keep": "the subject and its pose and position, the background and setting, "
                "every other object, the lighting and the camera viewpoint",
        "completeness": "Check every part of it separately — body, panels, padding, straps, "
                        "trims, handles, attachments. If any part still has the original's "
                        "colour or shape this is false; a leftover piece in the old colour is "
                        "the usual failure.",
        "instruction": "the product worn by, held by or attached to the subject",
        "preserve": "the subject itself and its exact pose and position, the background and "
                    "setting, the lighting, the camera viewpoint and the framing",
    },
    "subject": {
        "noun": "main subject",
        "label": "subject",
        # A subject fills much of the frame, so its masses legitimately shift.
        "layout_min": 0.30,
        "keep": "the background and setting, every other object, the lighting and the camera "
                "viewpoint, and the subject's POSE, POSITION and SIZE in frame",
        "completeness": "The subject in B must be recognisably the one from C — same species, "
                        "breed, colouring and markings — not the original merely restyled.",
        "instruction": "the main subject (the person or animal)",
        "preserve": "the subject's exact pose, position, size and orientation in frame, the "
                    "background and setting, any product it wears or holds, the lighting, the "
                    "camera viewpoint and the framing",
    },
    "environment": {
        "noun": "location",
        "label": "environment",
        # Replacing the setting redraws nearly every edge; structure cannot judge it.
        "layout_min": None,
        "keep": "the subject itself — same species, breed, colouring, pose, position and size "
                "in frame — any product it wears or holds, and the camera viewpoint and framing",
        "completeness": "The location in B must read as the KIND of place shown in C — its "
                        "surface, water, terrain, vegetation and light. It does NOT have to "
                        "match C's exact composition or viewpoint; C is a reference for the "
                        "place, not a backdrop to paste in. Also false if the result looks "
                        "composited: a cut-out subject, mismatched lighting or shadows, or "
                        "objects left over from the original location that do not belong in "
                        "the new one.",
        "instruction": "the location the action happens in",
        "preserve": "the subject exactly as it is — same pose, position, size, orientation and "
                    "appearance — along with any product it wears or holds, and the camera "
                    "viewpoint and framing",
    },
}

# Applied most-destructive first. Environment repaints nearly the whole frame,
# so anything swapped before it gets mangled by it — a product replaced first
# came back altered once the setting was redrawn around it. Product goes last:
# it is the smallest, most precisely verified edit, and nothing after it can
# disturb what it just fixed.
SWAP_ORDER = ["environment", "subject", "product"]


def _verify_swap(base_url, edited_url, ref_urls, slot="product", removed_props=None,
                 masked=False, parts=None, count=1):
    """Ask the vision model whether the swap did what was asked.

    Pixel heuristics cannot answer this. A whole-frame colour histogram scores
    an untouched vest and a recoloured one within 0.002 of each other, because
    the product is a small part of the frame. Layout is checkable in code;
    "is this now the right product, in the same scene" is not, and a Claude
    call costs cents against the ~$0.70 the video clip costs.

    Returns (scene_kept, product_swapped, note). Fails open on error — a broken
    check should not block generation.
    """
    try:
        import requests as _rq

        def block(url, label):
            raw = _rq.get(url, timeout=120).content
            return [
                {"type": "text", "text": label},
                {"type": "image", "source": {
                    "type": "base64", "media_type": _media_type(raw),
                    "data": base64.b64encode(raw).decode()}},
            ]

        spec = SWAP_SLOTS[slot]
        content = block(base_url, "IMAGE A — the original frame:")
        content += block(edited_url, "IMAGE B — the edited frame:")
        for i, u in enumerate(ref_urls[:3], start=1):
            content += block(u, f"IMAGE C{i} — the replacement {spec['noun']}:")
        if masked:
            # The subject was cut out and handed over as a mask, so those pixels
            # are preserved by construction. Asking a vision model to re-judge
            # the pose only produces false rejections — it reads the loss of a
            # prop the subject was leaning on as the subject having moved.
            content.append({"type": "text", "text":
                "B is A with the background repainted through a mask; the subject itself was "
                "protected and cannot have moved. Do not judge the subject's pose or position."
                + ("\n\nObjects deliberately removed with the old location: "
                   + "; ".join(removed_props[:6]) + ". Their absence is correct."
                   if removed_props else "") +
                "\n\nAnswer with JSON only, no prose:\n"
                '{"scene_kept": <true unless the subject itself was visibly altered or '
                'duplicated>, '
                '"swapped": <true if the location in B now reads as the kind of place shown in '
                'the C images AND the result looks like a real photograph taken there — light '
                'on the subject consistent with the new setting, plausible shadows and '
                'reflections, the subject sitting in the scene rather than pasted on top. '
                'false if it still shows the old location, or if it looks composited>, '
                '"note": "<one short sentence on what is wrong, if anything>"}'})
            resp = client.messages.create(model=MODEL, max_tokens=400,
                                          messages=[{"role": "user", "content": content}])
            text = "".join(b.text for b in resp.content if b.type == "text")
            data = _extract_json(text)
            return (bool(data.get("scene_kept")), bool(data.get("swapped")),
                    str(data.get("note") or ""))

        props_note = ""
        if removed_props:
            props_note = (
                "\n\nThese objects belonged to A's original location and were DELIBERATELY "
                "removed: " + "; ".join(removed_props[:6]) + ". Their absence from B is "
                "CORRECT and must not count against it.")

        content.append({"type": "text", "text":
            f"B was meant to be A with only the {spec['noun']} replaced by the one in the C "
            "images.\n\n"
            "Judge the CONTENT of the scene, not the file. Ignore resolution, image size, "
            "compression and small crop differences at the edges — those are expected and "
            "harmless.\n\n"
            "Every one of these models repaints the WHOLE frame; it cannot edit a region in "
            "isolation. So background texture WILL come back slightly different — foliage with "
            "different leaves, a fence with different grain, softer or sharper detail, small "
            "shifts in tone. That is the cost of the edit, not a failure, and rejecting it "
            "sends back the untouched original instead, which is a worse outcome. Judge whether "
            "it is still the SAME SHOT: same place, same subjects, same number of them, same "
            "poses and positions, same camera angle and framing, same time of day and lighting "
            "mood." + props_note + "\n\n"
            "Answer with JSON only, no prose:\n"
            '{"scene_kept": <true if everything OUTSIDE the replaced ' + spec['noun'] + ' is '
            'recognisably the same: ' + spec['keep'] + '. false only if it is genuinely a '
            'different shot — a subject moved, was added, removed or changed identity, the '
            'setting became a different place, or the camera moved. Re-rendered background '
            'detail alone is NOT false>, '
            '"swapped": <true ONLY if the ' + spec['noun'] + ' in B has been COMPLETELY '
            'replaced by the one in C. ' + spec['completeness']
            + (f" A has {count} SEPARATE copies of the product, one per subject. Count them in "
               f"B and check EACH ONE: all {count} must now be the C product. If even one is "
               "still the original, this is false — say which. Also false if B is a grid, "
               "collage or set of variations rather than the single scene from A."
               if count and count > 1 else "")
            + (" Go through these parts ONE BY ONE and confirm each has taken the C "
               "product's colour and form: " + "; ".join(parts[:8]) + "."
               if parts else "") + '>, '
            '"note": "<one short sentence on what differs; if the swap is partial, name what '
            'was missed>"}'})

        resp = client.messages.create(model=MODEL, max_tokens=400,
                                      messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        data = _extract_json(text)
        return (bool(data.get("scene_kept")), bool(data.get("swapped")),
                str(data.get("note") or ""))
    except Exception as e:
        # Still fails open — a broken checker must not block generation — but it
        # says what broke. The silent version hid a hard 400 behind a status line
        # that read like a pass.
        return True, True, f"(verification unavailable: {type(e).__name__}: {e})"


def _encode_frame(frame, max_side=768, quality=82):
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def detect_shots(video_bytes, min_shot=0.5, cut_threshold=0.55):
    """Split the reference into its real shots, at its real cut points.

    Sampling 16 evenly-spaced frames tells you nothing about where the editor
    actually cut, and produces "scenes" that do not exist in the source — an
    8s single-take clip came back as four invented scenes. Cuts are found by
    correlating consecutive HSV histograms: a hard cut drops the correlation
    sharply, while motion within a shot does not.

    Returns [{index, start, end, duration, first_b64, mid_b64, last_b64}].
    A clip with no cuts yields exactly one shot covering the whole thing.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or fps <= 0:
            cap.release()
            return []

        # Decode straight through and histogram every Nth frame. Seeking to
        # each sample instead would mean thousands of keyframe seeks on a long
        # clip, which takes minutes; sequential decode takes seconds.
        step = max(1, int(round(fps / 10)))
        boundaries, prev_hist, idx = [0], None, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                if prev_hist is not None:
                    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if corr < cut_threshold and (idx - boundaries[-1]) / fps >= min_shot:
                        boundaries.append(idx)
                prev_hist = hist
            idx += 1
        total = max(total, idx)
        boundaries.append(total)

        shots = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            if (b - a) / fps < min_shot and shots:
                continue
            picks = {}
            for name, pos in (("first", a + max(1, (b - a) // 12)),
                              ("mid", (a + b) // 2),
                              ("last", max(a, b - max(2, (b - a) // 12)))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos, total - 1))
                ok, frame = cap.read()
                picks[name] = _encode_frame(frame) if ok else None
            if not picks["first"]:
                continue
            shots.append({
                "index": len(shots) + 1,
                "start": round(a / fps, 2),
                "end": round(b / fps, 2),
                "duration": round((b - a) / fps, 2),
                "first_b64": picks["first"],
                "mid_b64": picks["mid"],
                "last_b64": picks["last"],
            })
        cap.release()
        return shots
    except Exception:
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def detect_overlay_regions(shots, tolerance=10, pad=10):
    """Find caption text burned into the reference's pixels.

    A reference ad almost always carries a hook caption ("watch my nervous
    dachshund…") baked into the footage. It arrives in every opening frame we
    hand to the video model, which then tries to *redraw* those letters — they
    smear, wobble and dissolve across the clip, and that artefact alone is
    enough to make a recreation read as AI-generated.

    Detection is free and needs no OCR: an overlay is by definition the part of
    the picture that does NOT change when the editor cuts to a different shot.
    Comparing frames from different shots, pixels whose value never moves are
    either pasted-on graphics or flat bars, and the two are told apart by
    texture — lettering has strong local contrast, a letterbox bar has none.

    Returns {"boxes": [(x, y, w, h)], "size": (w, h), "preview_b64": str,
    "mask_png": bytes} or None. `mask_png` follows the gpt-image-2 convention
    used for the environment swap: transparent = the part it may repaint.
    """
    try:
        import numpy as np

        frames, shot_ids = [], set()
        for s in shots:
            for name in ("first_b64", "mid_b64", "last_b64"):
                if not s.get(name):
                    continue
                arr = cv2.imdecode(
                    np.frombuffer(base64.b64decode(s[name]), np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    frames.append((s["index"], arr))
                    shot_ids.add(s["index"])
        if len(frames) < 4:
            return None
        h, w = frames[0][1].shape[:2]
        frames = [(i, f) for i, f in frames if f.shape[:2] == (h, w)]
        if len(frames) < 4:
            return None

        grey = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in frames]).astype(np.int16)
        spread = (grey.max(axis=0) - grey.min(axis=0)).astype(np.uint8)
        static = spread <= tolerance

        # With only one shot there is no cut to compare across, so a locked-off
        # camera makes the whole frame "static" and every guess is a false one.
        # Require most of the picture to have actually moved before believing
        # that the still parts are pasted on.
        if static.mean() > 0.55 or len(shot_ids) < 2:
            return None

        median = np.median(grey, axis=0).astype(np.uint8)
        edges = np.abs(cv2.Laplacian(median, cv2.CV_32F, ksize=3))

        # Glyph strokes are two or three pixels wide, so the usual
        # opening-to-remove-speckle erases the very thing being looked for.
        # Density does the same job without touching the strokes: a static
        # pixel counts only where its neighbourhood is static too, which is
        # true along lettering and not true of scattered coincidences in the
        # picture.
        density = cv2.boxFilter(static.astype(np.float32), -1, (9, 9), normalize=True)
        mask = ((static & (density > 0.05)).astype(np.uint8) * 255)
        # Letters are separate blobs; joining them sideways turns a line of text
        # into one region instead of thirty.
        mask = cv2.dilate(mask, np.ones((5, 21), np.uint8), iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        frame_area = float(h * w)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if not (0.0008 * frame_area <= area <= 0.15 * frame_area):
                continue
            region = labels[y:y + bh, x:x + bw] == i
            if region.sum() < 40:
                continue
            # Flat regions — black bars, blown sky, a uniform backdrop — are
            # static too. Text is not flat: it is bright glyphs on something
            # darker, with hard edges.
            if float(median[y:y + bh, x:x + bw][region].std()) < 18:
                continue
            if float(edges[y:y + bh, x:x + bw][region].mean()) < 6:
                continue
            boxes.append((max(0, x - pad), max(0, y - pad),
                          min(w, x + bw + pad) - max(0, x - pad),
                          min(h, y + bh + pad) - max(0, y - pad)))

        if not boxes:
            return None
        if sum(b[2] * b[3] for b in boxes) > 0.25 * frame_area:
            return None    # too much of the picture to be an overlay

        # Opaque everywhere, transparent over the text: exactly what the masked
        # edit model treats as "you may repaint only this".
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[:, :, :3] = frames[0][1]
        rgba[:, :, 3] = 255
        preview = frames[0][1].copy()
        for x, y, bw, bh in boxes:
            rgba[y:y + bh, x:x + bw, 3] = 0
            cv2.rectangle(preview, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        ok_mask, mask_buf = cv2.imencode(".png", rgba)
        ok_prev, prev_buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not (ok_mask and ok_prev):
            return None
        return {
            "boxes": [list(map(int, b)) for b in boxes],
            "size": [int(w), int(h)],
            "mask_png": mask_buf.tobytes(),
            "preview_b64": base64.b64encode(prev_buf.tobytes()).decode(),
        }
    except Exception:
        return None


def transcribe(video_bytes):
    """Whisper transcription of the reference audio. Returns '' when there is no
    usable audio track or no OpenAI key — a silent reference is still analysable."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ""

    audio_path = None
    video_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name
        audio_path = video_path + ".mp3"

        # Strip to mono 16kHz mp3 — Whisper's upload limit is 25MB and raw
        # video easily blows past it.
        cmd = [
            video_assembler.ffmpeg_exe(), "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", audio_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            return ""

        from openai import OpenAI
        oa = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as f:
            result = oa.audio.transcriptions.create(model="whisper-1", file=f)
        return (result.text or "").strip()
    except Exception:
        return ""
    finally:
        for p in (audio_path, video_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# ── Analysis prompt ────────────────────────────────────────────────────────

RECREATE_PROMPT = """You are reconstructing a short-form video ad shot for shot.

The reference has been cut into its REAL shots at its REAL cut points. For each shot you get three
frames — its opening, middle and closing frame — labelled with the shot number and its measured
duration.

You are NOT designing a new ad. Each shot will be regenerated starting from ITS OWN OPENING FRAME
of the reference, then animated. Your job is to say what ACTUALLY HAPPENS inside each shot so the
motion can be reproduced. Do not invent new staging, new locations, new actions or extra shots.

{product_block}

## REFERENCE METADATA
{metadata}

## REFERENCE TRANSCRIPT
{transcript}

## EXTRA INSTRUCTIONS FROM THE USER
{notes}

## WHAT TO PRODUCE

Return ONE JSON object inside a ```json fenced block. No prose outside the block.

{{
  "title": "short name for this video project",
  "format_analysis": "2-3 sentences: what happens in this ad and why it works",
  "product_identity": "one precise sentence naming what the product physically is, including its distinguishing visual features. Read it off the frames.",
  "product_parts": ["every separately-coloured or separately-shaped part of the product, named as it would be seen — e.g. 'main body panel', 'chin-rest float', 'fin spikes', 'buckle straps', 'grab handle'. This list is used to make sure a product swap replaces ALL of it, not just the largest part."],
  "location_props": ["objects visible in the reference that belong to its LOCATION rather than to the subject or the product — e.g. 'inflatable paddleboard', 'wooden jetty', 'beach umbrella', 'kitchen counter'. Exclude the product and anything the subject wears or holds. If the location is later changed these have to be removed, so name them even when they seem incidental."],
  "environment_brief": "<only if NEW LOCATION photos were attached> one or two sentences describing the kind of place the action moves to — ground/surface, water, terrain, vegetation, structures, time of day, quality of light. Otherwise \\"\\".",
  "burned_in_text": <true/false — only meaningful if an OVERLAY CANDIDATE image is attached at the end. It marks in red the pixels that are identical in every shot of the reference. Answer true if those regions are lettering or graphics PASTED ON TOP of the footage — a hook caption, subtitles, a watermark, a logo bug, a sticker. Answer false if they are part of the filmed scene itself, or if no OVERLAY CANDIDATE image was attached. Getting this wrong in either direction is costly: a false positive erases real scenery, a false negative leaves text that the video model will smear into unreadable mush.>,
  "soundscape": "the diegetic sound this footage would really have, as a comma-separated list of sources, ordered loudest first — e.g. 'gentle water lapping against an inflatable board, light sea breeze, distant seabirds, occasional small splash'. Describe only sound that the pictured place and action would actually make. No music. No speech.",
  "voice": "<one of: Aria, Roger, Sarah, Laura, Charlie, George, Callum, River, Liam, Charlotte, Alice, Matilda, Will, Jessica, Eric, Chris, Brian, Daniel, Lily, Bill>",
  "scenes": [
    {{
      "index": <shot number, matching the frames above — one entry per shot, no more, no fewer>,
      "shot_description": "what is in this shot, one sentence",
      "motion_prompt": "WHAT MOVES between the opening and closing frame of THIS shot. Compare the three frames and describe the actual change: which way the subject moves, which way the camera moves, what enters or leaves frame. If the frames are nearly identical, say so — e.g. 'the dog stays still, floating gently; almost no camera movement'. Never invent motion that is not evidenced by the frames. Then, in the SAME string, add one short physical-contact sentence — see PHYSICAL GROUNDING below. It is part of this field, not a separate one.",
      "shows_product": <true/false — is the product visible in this shot>,
      "product_count": <how many SEPARATE copies of the product are visible in this shot. Count the wearers/holders: three dogs each in their own life jacket is 3, not 1. 0 if the product is not in this shot. This is used to make a swap replace EVERY copy — the usual failure is that only the most prominent one is changed.>,
      "voiceover": "what is said over this shot, taken from the transcript. \\"\\" if nothing is said.",
      "on_screen_text": "text actually visible on screen in this shot, or \\"\\" if none"
    }}
  ]
}}

## HARD RULES

1. ONE SCENE PER DETECTED SHOT. Exactly {shot_count}. Do not split, merge, add or drop shots. Do not
   set durations — they are measured from the video and will be applied for you.
2. MOTION MUST BE READ OFF THE FRAMES, not imagined. A static shot must stay static. This is the
   single most important rule: invented motion is what makes a recreation look like a different
   video.
3. DESCRIBE, DO NOT REDESIGN. Same location, same subject, same action, same camera. No new angles,
   no added props, no "improvements".
4. VOICEOVER COMES FROM THE TRANSCRIPT. Split the transcript across shots by what is said when. If
   the reference is silent, every voiceover is "".
5. WRITE EVERYTHING IN ENGLISH.

## PHYSICAL GROUNDING — END EVERY `motion_prompt` WITH THIS

The video model is given your `motion_prompt` and one still frame. It does not know what is solid,
what floats, what bears weight or what is holding still. Left unsaid it guesses, and it guesses
badly: paws sink through a floating mat, a dog that "walks toward the water" walks ON the water,
and a floating platform nobody described as stationary drifts off into the distance like it is
flying away. Every one of those came from a motion_prompt that described the action correctly and
said nothing about the physics.

So finish each `motion_prompt` with one sentence, in plain words, covering whichever apply:

- **What carries the subject's weight**, and that it stays solid — "the mat stays rigid and level
  under the dog, taking its weight, and the paws rest ON its surface, never sinking into or
  through it".
- **What is stationary**, named — "the yellow mat, the boat and the shoreline stay exactly where
  they are in frame; only the dog moves". Anything the subject is not carrying is stationary unless
  the frames show otherwise. Say it even when it feels obvious.
- **Where the waterline sits** on a swimming subject — "the dog is IN the water, chest-deep, body
  submerged with only its head and the top of the jacket above the surface, legs paddling under
  the water, never standing on top of it".
- **Contact through a transition** — if the subject enters water, say it displaces water and sinks
  in before surfacing, rather than landing on top of it.

Do NOT use "walks", "steps" or "runs" for anything happening in or on water — those words make the
model put the animal on the surface. Say "paddles", "swims", "pushes off", "slides in".

Now analyse the shots and return the JSON."""


RECREATE_SAME_PRODUCT = """## THE PRODUCT

Keep the product exactly as it appears in the reference — this is a reconstruction of the same ad
for the same product. Nothing about it changes."""


RECREATE_SWAP_PRODUCT = """## THE PRODUCT — IT IS BEING SWAPPED

The reference shows one product; the new video must show a DIFFERENT one, attached above as YOUR
PRODUCT photos. Everything else stays identical — same scene, same subject, same action, same
camera. Only the product itself is replaced.

Describe YOUR PRODUCT (from its photos, not the reference's) in `product_identity`.

A shot may show the product MORE THAN ONCE — several subjects each wearing or holding their own
copy, and often in different colours from each other. Count the copies per shot in
`product_count`. Every copy gets replaced, so an undercount leaves the reference's product visibly
in the finished video.

{product}"""


MODE_SAME_PRODUCT = """## WHAT THIS VIDEO IS FOR

Build a NEW ad for **the exact same product that appears in the reference video**. Same physical
object, same brand, same colours and shape. This is a fresh ad for that product — new shots, new
angles, new scenes — not a copy of the reference's specific footage.

Identify the product from the frames as precisely as you can, and report it in `product_identity`.

Every shot that shows the product will be rendered by an image model that receives REAL FRAMES of
the product lifted from this very video. So write `image_prompt` as a STAGING INSTRUCTION for those
frames, not as a description of the product itself:

  GOOD: "Place this exact product on wet timber decking beside a lake at golden hour, low three-
         quarter angle, shallow depth of field, water droplets on the surface."
  BAD:  "An orange dog life vest with a dinosaur fin on a dock."   <- re-describing it invites the
        model to invent a different one; the frames already show what it looks like."""


MODE_MY_PRODUCT = """## WHAT THIS VIDEO IS FOR

Reverse-engineer the reference's FORMAT — pacing, shot grammar, hook structure, on-screen text
rhythm — and apply it to a DIFFERENT product, given below. You are NOT copying the reference's
subject, only how it is constructed.

## THE PRODUCT THIS NEW VIDEO IS FOR
{product}

Photos of this product are attached above, labelled YOUR PRODUCT. Describe it in `product_identity`
from those photos, not from the reference video's product.

Shots that show the product will be rendered from those REAL photos, so write `image_prompt` as a
STAGING INSTRUCTION for them, not as a description of the product itself:

  GOOD: "Place this exact product on a marble bathroom counter, soft window light, close three-
         quarter angle, shallow depth of field."
  BAD:  "A white ceramic jar of face cream on a counter."   <- re-describing it invites the model
        to invent a different one; the photos already show what it looks like."""


ANALYSIS_PROMPT = """You are a short-form video ad director. You are given frames from a REFERENCE
video ad (in chronological order, each labelled with its timestamp and a FRAME INDEX), its
transcript, and technical metadata.

{mode_block}

## REFERENCE METADATA
{metadata}

## TARGET LENGTH
{target}

## REFERENCE TRANSCRIPT
{transcript}

## EXTRA INSTRUCTIONS FROM THE USER
{notes}

## WHAT TO PRODUCE

Return ONE JSON object inside a ```json fenced block. No prose outside the block.

{{
  "title": "short name for this video project",
  "format_analysis": "2-4 sentences: what format is this, why does it work, what is the hook mechanism",
  "product_identity": "one precise sentence naming what the product physically is, including its distinguishing visual features (colour, shape, markings). Read it off the frames.",
  "product_reference_frames": [<frame indices, 2 to 4 of them, whose frames show the PRODUCT most clearly and unobstructed — prefer close, well-lit, front-or-three-quarter views. These exact frames get fed to the image model as ground truth for what the product looks like.>],
  "total_duration": <number, seconds — MUST equal TARGET LENGTH below, and MUST equal the sum of your scene durations>,
  "voice": "<one of: Aria, Roger, Sarah, Laura, Charlie, George, Callum, River, Liam, Charlotte, Alice, Matilda, Will, Jessica, Eric, Chris, Brian, Daniel, Lily, Bill>",
  "scenes": [
    {{
      "index": 1,
      "duration": <number, seconds>,
      "kind": "broll" | "avatar",
      "shot_description": "what the viewer sees, one sentence",
      "shows_product": <broll only, true/false> true if the product is visible in this shot at all,
      "image_prompt": "<broll only> the FIRST FRAME of this shot. When shows_product is true this is a STAGING instruction applied to the real product reference images (see above) — say where the product is, the angle, lens, lighting, background, and what else is in frame, but do NOT re-describe the product's own appearance. When shows_product is false it is an ordinary text-to-image prompt for a shot the product is not in.",
      "motion_prompt": "<broll only> what MOVES in this shot — camera move and subject motion. Keep it to one clear movement; image-to-video models degrade when asked for several.",
      "avatar_line": "<avatar only> the exact words the person says on camera",
      "voiceover": "<broll only> narration over this shot, or \\"\\" for silence",
      "on_screen_text": "text burned on screen, or \\"\\" for none"
    }}
  ]
}}

## HARD RULES

1. SCENE COUNT AND TIMING MIRROR THE REFERENCE. If the reference cuts every 1.5s, your scenes are
   1.5s. If it holds a 6s shot, hold 6s. Read the cut rhythm off the frames — do not default to
   uniform scene lengths, which is the single most common way this goes wrong. Your scene durations
   MUST add up to TARGET LENGTH: if the target is shorter than the reference, keep the cut RHYTHM
   and use fewer scenes; if longer, add scenes. Never stretch scenes to pad the runtime.
2. THE FIRST SCENE IS THE HOOK. Whatever mechanism the reference uses in its first 3 seconds
   (question, bold claim, visual pattern-break, problem shown), use the SAME mechanism with the new
   product's angle.
3. "avatar" IS ONLY FOR SHOTS WHERE A PERSON TALKS TO CAMERA. Everything else is "broll". If the
   reference is pure b-roll with voiceover, produce zero avatar scenes. If it is a UGC creator
   talking with cutaways, mirror that mix exactly.
4. WRITE THE REFERENCE'S REAL SHOT LENGTHS. There is no minimum. If a shot holds for 1.2 seconds,
   write 1.2. Video models have their own minimum clip length, but that is handled downstream by
   generating a longer clip and trimming it — it is NOT a floor on what you write here, and padding
   every scene out to a uniform 4 seconds is a bug, not a safe default. Only 0.8s is too short.
5. IMAGE PROMPTS MUST NOT NAME REAL BRANDS, celebrities, or copyrighted characters, and must not
   describe the reference video's specific actors. Describe generic people by role and appearance.
   This does NOT apply to the product itself — keeping the product identical is the whole point.
6. VOICEOVER + AVATAR LINES TOGETHER MUST BE SPEAKABLE IN THE SCENE'S DURATION. Roughly 2.5 words
   per second. A 3-second scene gets ~7 words, not a sentence.
7. WRITE THE NARRATION AS ONE PERSON TALKING, NOT AS CAPTIONS. Read every scene's `voiceover` in
   order: together they must form connected, natural speech, because they are recorded as a single
   continuous take. Use contractions, everyday word order, and sentences that run across scene
   boundaries where that is how someone would actually say it. Never write clipped label-like
   fragments ("Two gummies. Twelve vitamins. No sugar.") — that is what makes it sound synthetic.
8. THE PRODUCT NEVER CHANGES. Its colour, shape, markings and materials are fixed across every
   scene. Do not have it appear in a different colourway, size or variant partway through, and do
   not mention alternatives.
7. ON-SCREEN TEXT IS SHORT. Under 6 words. Only where the reference actually shows text.
8. WRITE EVERYTHING IN ENGLISH.

Now analyse the frames and return the JSON."""


REWRITE_PROMPT = """Here is an existing video recipe JSON and a change request from the user.

Apply ONLY the requested change. Keep every other field byte-identical. Return the full updated
JSON object in a ```json fenced block, no prose outside it.

`duration` normally comes from the reference and must not be touched — but if the user explicitly
asks for a scene to be longer or shorter ("make the last shot 4 seconds", "hold on the product a
bit"), change that scene's `duration` to what they asked for. Leave the other scenes' durations
alone; the video simply gets longer or shorter.

## CURRENT RECIPE
```json
{recipe}
```

## CHANGE REQUEST
{instructions}"""


def _format_product(product, uploaded_count=0):
    if not product:
        if uploaded_count:
            return (f"See the {uploaded_count} attached YOUR PRODUCT photo(s) — that is the product. "
                    "No other details were given, so read everything off the photos.")
        return "(no product details given — write the recipe generically for a direct-response ecommerce product)"
    parts = [f"Name: {product.get('title', '(untitled)')}"]
    if product.get("price"):
        parts.append(f"Price: {product['price']}")
    if product.get("description"):
        desc = re.sub(r"<[^>]+>", " ", product["description"])
        desc = re.sub(r"\s+", " ", desc).strip()
        parts.append(f"Description: {desc[:1500]}")
    if product.get("url"):
        parts.append(f"URL: {product['url']}")
    return "\n".join(parts)


def _extract_json(text):
    m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    raw = m.group(1) if m else text
    raw = raw.strip()
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Model did not return a JSON object")
        raw = raw[start:end + 1]
    return json.loads(raw)


MIN_SCENE = 0.8

# Laplacian-correlation floor below which a "product swap" has really redrawn
# the whole shot rather than edited it.
COMPOSITION_MIN = 0.45

# Below this, the reference had no real narration — a stray name or two — and
# reading it aloud produces an obviously synthetic voice over silent footage.
NARRATION_MIN_WORDS = 6


# Words that show the prompt already pins down contact, weight and what holds
# still. If none appear, the model wrote a pure description of the action and
# left the physics to the video model, which is what sinks paws through mats
# and floats props away into the distance.
_GROUNDING_HINTS = (
    "stationary", "stays fixed", "stays put", "stays in place", "does not move",
    "doesn't move", "without drifting", "bearing", "bears the", "takes its weight",
    "waterline", "buoyant", "rigid", "supports the", "no solid ground",
    "remains still", "holds still", "fixed in one spot",
)


MISSING_SHOT_PROMPT = """The reference video was cut into {total} shots. A previous pass described all of
them except shot(s) {missing} — those entries are simply absent, so those shots would be animated
from an EMPTY prompt, which makes the video model improvise something that is not in the reference.

Below are the frames of ONLY the missing shot(s), each labelled with its shot number.

## REFERENCE TRANSCRIPT
{transcript}

Return ONE JSON object inside a ```json fenced block, no prose outside it, with one entry per
missing shot and nothing else:

{{
  "scenes": [
    {{
      "index": <the shot number exactly as labelled above>,
      "shot_description": "what is in this shot, one sentence",
      "motion_prompt": "WHAT MOVES between the opening and closing frame of THIS shot — read it off the frames, never invent motion. Then, in the SAME string, add one short physical-contact sentence: what carries the weight and that it is rigid, which objects are stationary, where the waterline sits. Never say walks/steps/runs for movement through water.",
      "shows_product": <true/false — is the product visible in this shot>,
      "product_count": <how many SEPARATE copies of the product are visible; count the wearers>,
      "voiceover": "what is said over this shot, taken from the transcript, or \\"\\"",
      "on_screen_text": "text visible on screen in this shot, or \\"\\""
    }}
  ]
}}

Describe only what the frames show. Do not redesign, do not add shots, write in English."""


def _describe_missing_shots(shots, described, transcript):
    """Fill in shots the analysis pass skipped.

    The analysis prompt says "exactly {shot_count}, no more, no fewer" and
    usually obeys — but on project 21 it returned three entries for four shots.
    The pairing loop below defaulted the fourth to empty strings, so its
    `motion_prompt` reached the (paid) video model as "" and the model invented
    a shot of its own; the swap for that scene was likewise asked for with no
    scene context at all. One skipped entry silently wrecked both.

    Rather than re-word the instruction and hope, ask again for just the ones
    that are missing, with only their frames attached. Returns the entries it
    managed to get, keyed by index — never raises.
    """
    missing = [s for s in shots if s["index"] not in described]
    if not missing:
        return {}

    content = []
    for s in missing:
        content.append({"type": "text",
                        "text": f"SHOT {s['index']} — {s['duration']}s "
                                f"({s['start']}s to {s['end']}s). Opening frame:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": s["first_b64"]}})
        for name in ("mid_b64", "last_b64"):
            if s.get(name):
                content.append({"type": "text",
                                "text": f"SHOT {s['index']} — {name.split('_')[0]} frame:"})
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": s[name]}})
    content.append({"type": "text", "text": MISSING_SHOT_PROMPT.format(
        total=len(shots),
        missing=", ".join(str(s["index"]) for s in missing),
        transcript=transcript or "(silent)",
    )})

    try:
        resp = client.messages.create(model=MODEL, max_tokens=4000,
                                      messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        recovered = _extract_json(text).get("scenes") or []
    except Exception:
        return {}

    wanted = {s["index"] for s in missing}
    return {d["index"]: d for d in recovered
            if isinstance(d, dict) and d.get("index") in wanted}


# What to animate a shot from when nothing could be read off its frames. It
# holds the framing and lets only what is already moving continue: anything
# richer would be invention, which is precisely what the empty prompt caused.
HOLD_STILL_MOTION = (
    "Continue this exact moment as filmed. The framing, the subject and everything "
    "else in the shot carry on as they are, with only the small natural movement "
    "already present — a slight drift of the camera and the subject's own gentle "
    "motion. Introduce no new action, and change nothing about what the subject "
    "is wearing, holding or standing on")


def _motion_for(scene, props=None):
    """The prompt a shot gets animated from. Never empty.

    A blank prompt is not a mild degradation — the video model is handed one
    frame and no instruction, so it invents its own shot. Falling back to a
    hold-still instruction keeps the reference's own frame in charge.
    """
    text = (scene.get("motion_prompt") or scene.get("shot_description") or "").strip()
    return _ground_motion_prompt(text or HOLD_STILL_MOTION, props)


def _ground_motion_prompt(text, props=None):
    """Append a physics sentence when the prompt has none.

    The analysis prompt asks for one, and usually gets it — but not reliably:
    the same request produced a grounded prompt on one run and "the dog walks
    toward the water" on the next. Rather than re-word the instruction and hope,
    anything that comes back without grounding gets a generic one appended. It
    is weaker than a specific sentence, but it is never absent.
    """
    body = (text or "").strip()
    if not body:
        return body
    low = body.lower()
    if any(h in low for h in _GROUNDING_HINTS):
        return body

    named = [p for p in (props or []) if isinstance(p, str) and p.strip()][:4]
    if named:
        listed = ", ".join(p.strip() for p in named)
        fixed = (f"The {listed} stay exactly where they are in frame and do not drift, "
                 "slide or recede into the distance")
    else:
        fixed = ("Everything the subject is not carrying stays exactly where it is in "
                 "frame and does not drift, slide or recede into the distance")
    return (f"{body.rstrip('. ')}. {fixed}; any surface the subject rests on is rigid and "
            "bears its weight without the feet sinking through it; anything in water is "
            "IN the water at a believable waterline, never standing or walking on the "
            "surface.")


def _as_count(value):
    """A scene's product_count, normalised. 1 when the model didn't say.

    Older recipes predate the field, and 1 is what the whole swap path assumed
    before it existed, so that is the safe default.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 1
    return max(0, min(n, 12))


def _instances_clause(count):
    """Spell out that several copies of the product are in frame.

    Everything here used to be phrased in the singular — "the product worn by
    the subject", "keep the subject in the same pose". Handed a frame with
    three dogs in three different life jackets, the edit models either changed
    only the front one or abandoned the frame entirely and returned a collage
    of three variations of a single dog. Naming the count, and demanding one
    photograph back, is what makes the edit stay inside the original shot.
    """
    if count is None or count < 2:
        return ""
    return (f"\n\nIMPORTANT — there are {count} SEPARATE copies of the product in this "
            f"photograph, worn or held by {count} different subjects. ALL {count} must be "
            "replaced, including any that are partly hidden, further away, at a different "
            "angle, or in a different colour from each other. A frame where only the nearest "
            "or largest one has changed is a FAILURE. The subjects themselves — how many "
            "there are, which is which, their poses and positions — do not change.\n\n"
            "Return ONE single photograph of that same scene. Never a grid, a collage, a "
            "split image, a before/after pair, or several variations.")


def _parts_clause(parts):
    """Spell out every component of the product.

    Without this the model swaps the largest panel and leaves the rest — a
    green chin-rest float survived on an otherwise blue vest, because the model
    treats "the product" as the one obvious shape.
    """
    parts = [p for p in (parts or []) if isinstance(p, str) and p.strip()][:8]
    if not parts:
        return ""
    listed = "; ".join(p.strip() for p in parts)
    return (f"\n\nThe product is made of several distinct parts: {listed}. EVERY one of them "
            "must be replaced. No part of the original product may remain — check especially "
            "for a leftover piece in the original's colour, which is the usual failure.")


def _environment_prompts(ref_count, brief="", props=None):
    """Move the action to a new kind of place, believably.

    Not a background replacement. Pasting the subject onto a different photo
    gives the cut-out, mismatched-light look; what is wanted is the same moment
    re-shot on location somewhere of that character. The reference photos are
    a brief, not a backplate.

    `props` are objects tied to the ORIGINAL location. They have to be named
    for removal: they are neither subject nor product, so every "keep
    everything else" instruction silently preserves them, and a paddleboard
    followed a dog onto a beach that had none.
    """
    figs = "the reference photo" if ref_count == 1 else f"the {ref_count} reference photos"
    props = [p for p in (props or []) if isinstance(p, str) and p.strip()][:6]
    remove = ""
    if props:
        remove = ("\n\nThese belong to the ORIGINAL location and must be GONE from the result, "
                  "not carried over: " + "; ".join(p.strip() for p in props) +
                  ". Do not add replacements for them unless the new place would naturally "
                  "have them. Fill the space they occupied with whatever the new location "
                  "actually has there.")
    place = f"\n\nThe new location: {brief.strip()}" if brief and brief.strip() else ""

    common = (
        "Keep the subject EXACTLY as it is — same species, breed, colouring and markings, same "
        "pose, same position and size in the frame, same orientation — and keep any product it "
        "wears or holds unchanged. Keep the camera viewpoint, the crop and the framing.\n\n"
        "The result must look like a real photograph taken in that place: the light on the "
        "subject matches the new location's light, shadows and reflections fall correctly, the "
        "subject sits in the scene rather than on top of it, and depth of field is consistent. "
        "It must not look composited or cut out."
        + remove
    )
    return [
        (f"Re-shoot this exact moment on location somewhere like {figs}.{place}\n\n"
         f"Take the CHARACTER of the place from the reference — its surface, water, terrain, "
         f"vegetation, colours and quality of light. Do NOT copy its composition, its camera "
         f"angle or its specific objects; it is a mood reference, not a backdrop to paste in. "
         f"Invent the details that make the new setting coherent.\n\n" + common),
        (f"Change where this photograph was taken.{place}\n\n"
         f"Everything around the subject becomes a new setting in the spirit of {figs} — you "
         f"may reinterpret it freely, use only part of it, or extend it, as long as the result "
         f"is a believable real place of that kind seen from this camera position.\n\n"
         + common),
    ]


def _continuity_clause(slot):
    """Tell the model the last image is continuity, not a second thing to copy.

    Without it the frame handed back from the previous shot reads as another
    reference to blend with, and the model starts borrowing its composition.
    """
    spec = SWAP_SLOTS[slot]
    return (
        f"\n\nCONTINUITY — READ THIS ABOUT THE LAST ATTACHED IMAGE.\n"
        f"The FINAL image is not a new thing to put in the shot. It is a frame from an "
        f"EARLIER SHOT OF THIS SAME VIDEO, in which the {spec['noun']} was already replaced "
        f"and accepted. The {spec['noun']} you produce must be the SAME ONE seen there — "
        f"same identity, same colours, same markings, same materials and proportions — so "
        f"that the finished video does not visibly change it halfway through.\n"
        f"Take ONLY appearance from that frame. Ignore its composition, its camera angle, "
        f"its pose and everything else in it; the shot being edited is Figure 1 and only "
        f"Figure 1."
    )


def _slot_prompts(slot, ref_count, parts=None, brief="", props=None, count=1,
                  continuity=False):
    """Escalating edit instructions for one swap slot, best first.

    Product and environment have their own hand-tuned wording — they are the
    two that took the most iterations. Subject is generated from the slot's
    description, being the straightforward case: replace one thing, leave
    everything named in `preserve` untouched.
    """
    if continuity:
        tail = _continuity_clause(slot)
        return [p + tail for p in _slot_prompts(slot, ref_count, parts, brief, props, count)]
    if slot == "product":
        return _swap_prompts(ref_count, parts, count)
    if slot == "environment":
        return _environment_prompts(ref_count, brief, props)

    spec = SWAP_SLOTS[slot]
    figs = ("Figure 2" if ref_count == 1
            else "Figures 2" + "".join(f", {i}" for i in range(3, ref_count + 2)))
    others = "the image that follows" if ref_count == 1 else f"the {ref_count} images that follow"
    return [
        (
            f"Figure 1 is a photograph. {figs} show {spec['instruction']} to use instead.\n\n"
            f"Edit Figure 1 so that {spec['instruction']} is replaced by the one from {figs}.\n\n"
            f"Keep unchanged: {spec['preserve']}. Do not restage the shot. Do not change what "
            f"is happening. Only {spec['noun']} changes."
        ),
        (
            "You are performing an inpainting edit, not generating a new picture.\n\n"
            f"IMAGE 1 is the ONLY image whose composition matters. Reproduce it, then repaint "
            f"ONLY {spec['instruction']}, replacing it with the one shown in {others}.\n\n"
            f"These must survive the edit untouched: {spec['preserve']}. The reference images "
            f"supply nothing but the new {spec['noun']} — ignore their composition, their "
            "camera angle and anything else in them."
        ),
    ]


def _swap_prompts(ref_count, parts=None, count=1):
    """Instructions for swapping a product into an existing frame, best first.

    Each pairs with a rung of fal_client.EDIT_LADDER. The first uses Seedream's
    Figure-referencing, which is built for "the thing in Figure 1 becomes the
    thing in Figure 2". The second frames it as inpainting, which is what makes
    Nano Banana actually perform the swap rather than quietly no-op.

    `count` is how many copies of the product are in the frame. Every phrase
    below has to agree with it: asked to replace "the product worn by the
    subject" in a shot holding three of them, the models changed one and left
    two, or gave up on the frame and returned a collage.
    """
    figs = ("Figure 2" if ref_count == 1
            else "Figures 2" + "".join(f", {i}" for i in range(3, ref_count + 2)))
    others = "the image that follows" if ref_count == 1 else f"the {ref_count} images that follow"
    many = count and count > 1
    # "the product worn by the subject" vs "each of the 3 products, one per subject"
    worn = (f"every one of the {count} products worn by or attached to the {count} subjects"
            if many else "the product worn by or attached to the subject")
    region = (f"the product regions — all {count} of them, one on each subject"
              if many else "the product region — the item worn by or attached to the subject")
    shown = (f"each of the {count} products worn by or shown with the subjects"
             if many else "the product worn by or shown with the subject")
    subj_pose = ("The subjects all stay in the same places in the same poses"
                 if many else "The subject stays in the same place in the same pose")
    keep_pose = ("Keep every subject in the same pose and the same position"
                 if many else "Keep the subject in the same pose and the same position")
    outside = ("the subjects and their exact poses and positions"
               if many else "the subject and its exact pose and position")
    pc = _parts_clause(parts) + _instances_clause(count)
    return [p + pc for p in [
        (
            f"Figure 1 is a photograph. {figs} show a product on its own.\n\n"
            f"Edit Figure 1 so that {worn} is replaced "
            f"by the product from {figs}, matching its real colour, shape and markings.\n\n"
            f"Change nothing else in Figure 1. {keep_pose}, "
            "keep the background, water, horizon and every other object exactly "
            "where they are, keep the lighting and the camera viewpoint, and keep the same "
            f"crop and framing. Do not use the background or setting from {figs} — those are "
            "product photos, and only the product itself should be taken from them."
        ),
        (
            "You are performing an inpainting edit, not generating a new picture.\n\n"
            "IMAGE 1 is the ONLY image whose scene matters. Reproduce IMAGE 1 exactly. Then "
            f"paint over ONLY {region}"
            f", replacing it with the product shown in {others}.\n\n"
            "Every pixel outside that product region must be identical to IMAGE 1. "
            f"{subj_pose}. The background, water depth, "
            "shoreline, sky and every other object stay exactly as they are. The camera does "
            "not move. The framing and crop do not change. The reference images contribute "
            "the product's appearance and NOTHING ELSE — ignore their backgrounds entirely.\n\n"
            "The replacement must be clearly visible: match the reference product's real "
            "colour, shape and markings, not the colour of the product already in IMAGE 1."
        ),
        (
            f"IMAGE 1 is a photograph to edit. {others.capitalize()} show a replacement "
            "product, on its own, for reference only.\n\n"
            f"Return IMAGE 1 with one single change: {shown} "
            "is replaced by the reference product, matching its real colour, shape "
            "and markings.\n\n"
            f"Everything else in IMAGE 1 must be unchanged — {outside}"
            ", the background, the water, the horizon, every other object, the "
            "lighting, the shadows, the camera angle, the crop and the framing. Do not move "
            "anything. Do not re-stage or re-imagine the scene. Do not borrow the background "
            "or camera angle from the reference images; they are product photos only."
        ),
    ]]


def _fit_to_target(scenes, target):
    """Scale scene durations so they sum to `target`, preserving the cut rhythm.

    The prompt asks for this, but the model reliably drifts — it padded every
    scene to a flat 4s on an 8s reference — so it is enforced here rather than
    hoped for. Proportional scaling keeps the relative pacing intact; only the
    0.8s floor can distort it, and then only for scenes already at the floor.
    """
    if not scenes or not target or target <= 0:
        return None

    current = sum(s["duration"] for s in scenes)
    if current <= 0:
        return None
    if abs(current - target) <= max(0.5, target * 0.05):
        return None      # close enough; leave the model's timing alone

    factor = target / current
    for s in scenes:
        s["duration"] = max(MIN_SCENE, round(s["duration"] * factor, 2))

    # The floor can push the sum back up; absorb the excess in the longest
    # scenes, which have the most slack before the pacing visibly changes.
    for _ in range(len(scenes)):
        drift = round(sum(s["duration"] for s in scenes) - target, 2)
        if abs(drift) <= 0.05:
            break
        adjustable = sorted((s for s in scenes if s["duration"] > MIN_SCENE),
                            key=lambda s: -s["duration"])
        if not adjustable:
            break
        head = adjustable[0]
        head["duration"] = max(MIN_SCENE, round(head["duration"] - drift, 2))

    return round(current, 2)


def _playable_windows(model_key):
    """Slot lengths a model can fill without visible retiming, as ranges.

    A model sells fixed clip lengths. For each length L the slots it can fill
    honestly run from L / MAX_SPEEDUP (compressed as far as the eye allows) to
    L * MAX_SLOWDOWN (eased out as far as the eye allows). Kling's 5s and 10s
    give [3.7, 5.75] and [7.41, 11.5] — note the gap between them, which is
    real: there is no way to fill a 6.5s slot from a 5s or 10s clip without
    something looking wrong.

    Returned sorted and rounded up/down inwards, so a value inside a window is
    genuinely inside it after floating point.
    """
    spec = fal_client.VIDEO_MODELS.get(model_key) or {}
    windows = []
    for d in sorted({float(x) for x in (spec.get("durations") or [5])}):
        lo = math.ceil(d / video_assembler.MAX_SPEEDUP * 100) / 100
        hi = math.floor(d * video_assembler.MAX_SLOWDOWN * 100) / 100
        if hi >= lo:
            windows.append((lo, hi))
    return windows or [(3.7, 5.75)]


def _playable_minimum(model_key):
    """Shortest slot this model can fill without the result looking retimed."""
    return _playable_windows(model_key)[0][0]


def _nearest_playable(duration, windows):
    """Smallest honest slot length that is at least `duration`.

    Slots only ever grow. Shrinking one to reach a nearer window would need the
    time back from somewhere else, and every scene it came from would then be
    wrong instead — the reference's pacing is the thing being preserved, so the
    cut gets longer rather than differently wrong.
    """
    for lo, hi in windows:
        if duration < lo:
            return lo
        if duration <= hi:
            return duration
    return windows[-1][1]


def _fit_slots_to_model(scenes, model_key):
    """Widen any slot the model cannot fill without visible retiming.

    This used to raise only the slots that were too SHORT for the model's
    briefest clip, and it kept the total length by taking the time back from the
    longest scenes. Both halves were wrong.

    Too-short was never the only failure: a 2.02s slot bought a 5s clip and
    played it at 2.48x, which reads as fast-forward, and a 5.7s slot bought a
    10s clip and played it at 1.75x. Slots between the model's clip lengths are
    just as unplayable as slots below the shortest one.

    And taking the time back from other scenes moved the problem rather than
    fixing it — every donor scene got closer to needing a speed-up of its own.
    The cut now grows instead, and the caller reports the new length: a longer
    video that moves at the reference's pace is what was actually asked for,
    where a video of exactly the requested length that fast-forwards is not.

    Returns [(index, old, new)] for reporting.
    """
    windows = _playable_windows(model_key)
    changed = []
    for s in scenes:
        if s.get("kind") != "broll":
            continue
        want = _nearest_playable(s["duration"], windows)
        # Durations carry two decimals and the window edges are rounded inwards
        # to match, so the tolerance has to be finer than one of those steps —
        # at 0.01 a 3.70s slot counted as already inside a window that starts at
        # 3.71, and stayed a hair over the speed cap.
        if want > s["duration"] + 0.005:
            changed.append((s["index"], s["duration"], want))
            s["duration"] = want
    return changed


def _normalise_recipe(recipe, meta, target_duration=None):
    """Fill in defaults and clamp anything the model may have got wrong."""
    recipe.setdefault("title", "Untitled video")
    recipe.setdefault("aspect_ratio", meta.get("aspect_ratio", "9:16"))
    if recipe["aspect_ratio"] not in ("9:16", "1:1", "16:9"):
        recipe["aspect_ratio"] = "9:16"
    if recipe.get("voice") not in fal_client.AVATAR_VOICES:
        recipe["voice"] = "Sarah"

    scenes = recipe.get("scenes") or []
    clean = []
    for i, s in enumerate(scenes, start=1):
        s["index"] = i
        try:
            s["duration"] = max(MIN_SCENE, round(float(s.get("duration", 4)), 2))
        except (TypeError, ValueError):
            s["duration"] = 4.0
        if s.get("kind") not in ("broll", "avatar"):
            s["kind"] = "broll"
        for field in ("shot_description", "image_prompt", "motion_prompt",
                      "avatar_line", "voiceover", "on_screen_text"):
            s.setdefault(field, "")
            if s[field] is None:
                s[field] = ""
        s["shows_product"] = bool(s.get("shows_product")) and s["kind"] == "broll"
        clean.append(s)
    recipe["scenes"] = clean
    was = _fit_to_target(clean, target_duration)
    if was is not None:
        recipe["duration_corrected_from"] = was
    recipe["total_duration"] = round(sum(s["duration"] for s in clean), 2)
    recipe.setdefault("product_identity", "")
    recipe.setdefault("product_reference_urls", [])
    return recipe


# ── Phase 1a: shot-for-shot reconstruction ─────────────────────────────────

def _analyze_recreate(video_bytes, product, notes, product_images, target_duration,
                      subject_images=None, environment_images=None):
    """Build a recipe that rebuilds the reference shot for shot.

    The difference from the other modes is where each shot's first frame comes
    from: here it is the reference's own opening frame for that shot, uploaded
    as-is, rather than an image invented from a text prompt. That is what keeps
    the subject, framing, location and action the same instead of merely
    similar.
    """
    try:
        yield {"type": "status", "text": "🎞️ Reading video…"}
        meta = probe_video(video_bytes)
        if meta["duration"] <= 0:
            yield {"type": "done", "error": "Could not read this video file. Try re-exporting it as MP4 (H.264)."}
            return
        yield {"type": "status",
               "text": f"🎞️ {meta['duration']}s · {meta['width']}×{meta['height']} · {meta['aspect_ratio']}"}

        yield {"type": "status", "text": "✂️ Finding the real cut points…"}
        shots = detect_shots(video_bytes)
        if not shots:
            yield {"type": "done", "error": "Could not read any shots from this video."}
            return
        yield {"type": "status", "text": (
            f"✂️ {len(shots)} shot(s): "
            + ", ".join(f"{s['duration']}s" for s in shots[:8])
            + (" …" if len(shots) > 8 else ""))}

        yield {"type": "status", "text": "🎙️ Transcribing audio…"}
        transcript = transcribe(video_bytes)
        yield {"type": "status", "text": (
            f"🎙️ Transcript: {len(transcript.split())} words" if transcript
            else "🎙️ No speech — the recreation will be silent too")}

        swapping = bool(product_images) or bool((product or {}).get("image"))
        yield {"type": "status", "text": "🧠 Reading what happens in each shot…"}

        content = []
        for s in shots:
            content.append({"type": "text",
                            "text": f"SHOT {s['index']} — {s['duration']}s "
                                    f"({s['start']}s to {s['end']}s). Opening frame:"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": s["first_b64"]}})
            for name in ("mid_b64", "last_b64"):
                if s.get(name):
                    content.append({"type": "text",
                                    "text": f"SHOT {s['index']} — {name.split('_')[0]} frame:"})
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": s[name]}})

        for i, img in enumerate(product_images, start=1):
            content.append({"type": "text", "text": f"YOUR PRODUCT — PHOTO {i}:"})
            content.append({"type": "image", "source": {
                "type": "base64",
                "media_type": img.get("media_type", "image/jpeg"),
                "data": img["b64"]}})

        for i, img in enumerate(environment_images or [], start=1):
            content.append({"type": "text", "text": f"NEW LOCATION — REFERENCE PHOTO {i}:"})
            content.append({"type": "image", "source": {
                "type": "base64",
                "media_type": img.get("media_type", "image/jpeg"),
                "data": img["b64"]}})
        if environment_images:
            content.append({"type": "text", "text":
                "The action is being moved to a place like the NEW LOCATION photos. Describe "
                "that place in `environment_brief`: surface or ground, water if any, terrain, "
                "vegetation, structures, time of day and quality of light. Describe the KIND "
                "of place, not that specific photograph's composition — it is a reference for "
                "the setting, not a backdrop to be pasted in.\n\n"
                "`soundscape` must then describe what the NEW location sounds like, not the "
                "original one."})

        # Free, pixel-level: what never moves between cuts is probably pasted on.
        # The model only has to say whether it is text, not find it.
        overlay = detect_overlay_regions(shots)
        if overlay:
            content.append({"type": "text", "text": (
                "OVERLAY CANDIDATE — the red boxes mark every pixel region that is IDENTICAL "
                "in all of the frames above, across different shots. Decide whether they are "
                "burned-in text/graphics pasted over the footage, and answer in `burned_in_text`.")})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": overlay["preview_b64"]}})

        product_block = (RECREATE_SWAP_PRODUCT.format(product=_format_product(product, len(product_images)))
                         if swapping else RECREATE_SAME_PRODUCT)
        content.append({"type": "text", "text": RECREATE_PROMPT.format(
            product_block=product_block,
            metadata=json.dumps(meta),
            transcript=transcript or "(silent)",
            notes=notes or "(none)",
            shot_count=len(shots),
        )})

        resp = client.messages.create(model=MODEL, max_tokens=8000,
                                      messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        recipe = _extract_json(text)

        # Pair the model's descriptions with the measured shots. Durations come
        # from the video, never from the model — it has no way to know them and
        # every past attempt to let it guess produced the wrong runtime.
        described = {s.get("index"): s for s in (recipe.get("scenes") or [])}

        # "Exactly {shot_count}, no more, no fewer" is not always obeyed. A shot
        # with no entry would be animated from an empty prompt and swapped with
        # no scene context, so ask again for just the ones that came back short.
        if any(shot["index"] not in described for shot in shots):
            gap = [shot["index"] for shot in shots if shot["index"] not in described]
            yield {"type": "status", "text": (
                f"🔁 Shot(s) {', '.join(str(i) for i in gap)} came back undescribed — reading them again…")}
            described.update(_describe_missing_shots(shots, described, transcript))
            still = [shot["index"] for shot in shots if shot["index"] not in described]
            if still:
                yield {"type": "status", "text": (
                    f"⚠️ Shot(s) {', '.join(str(i) for i in still)} could not be read — "
                    "they will be animated from their own frame with a hold-still prompt")}

        scenes = []
        for shot in shots:
            d = described.get(shot["index"], {})
            scenes.append({
                "index": shot["index"],
                "duration": shot["duration"],
                "kind": "broll",
                "shot_description": d.get("shot_description") or "",
                "motion_prompt": d.get("motion_prompt") or "",
                "image_prompt": "",
                "avatar_line": "",
                "shows_product": bool(d.get("shows_product", True)),
                "product_count": _as_count(d.get("product_count")),
                "voiceover": d.get("voiceover") or "",
                "on_screen_text": d.get("on_screen_text") or "",
                "source_start": shot["start"],
            })
        recipe["scenes"] = scenes

        recipe.setdefault("title", "Recreated video")
        recipe["aspect_ratio"] = meta["aspect_ratio"]
        if recipe.get("voice") not in fal_client.AVATAR_VOICES:
            recipe["voice"] = "Sarah"
        recipe["mode"] = "recreate"
        recipe["source_duration"] = meta["duration"]
        recipe.setdefault("product_identity", "")
        recipe.setdefault("product_parts", [])
        recipe.setdefault("location_props", [])
        recipe.setdefault("environment_brief", "")
        recipe.setdefault("soundscape", "")

        if target_duration and abs(float(target_duration) - meta["duration"]) > 0.5:
            was = _fit_to_target(scenes, float(target_duration))
            if was is not None:
                recipe["duration_corrected_from"] = was
                yield {"type": "status",
                       "text": f"⏱️ Reference is {was}s — scaled shots to your {target_duration}s target"}
        recipe["total_duration"] = round(sum(s["duration"] for s in scenes), 2)
        recipe["target_duration"] = recipe["total_duration"]

        # The reference's own caption is baked into every frame we are about to
        # use as a starting image. Left there, the video model repaints the
        # letters and they crawl and dissolve through the whole clip.
        recipe["overlay_boxes"] = []
        recipe["overlay_mask_url"] = ""
        if overlay and bool(recipe.get("burned_in_text")):
            try:
                recipe["overlay_mask_url"] = fal_client.upload_bytes(
                    overlay["mask_png"], "overlay_mask.png", "image/png")
                recipe["overlay_boxes"] = overlay["boxes"]
                yield {"type": "status", "text": (
                    f"🧽 Burned-in text found in the reference "
                    f"({len(overlay['boxes'])} region(s)) — it will be erased from each "
                    "starting frame before animation")}
            except Exception as e:
                yield {"type": "status",
                       "text": f"⚠️ Could not prepare the text-removal mask: {e}"}
        elif overlay:
            yield {"type": "status",
                   "text": "🧽 Static regions checked — not pasted-on text, left alone"}
        recipe.pop("burned_in_text", None)

        # Each shot's own opening frame is the visual base for regenerating it.
        yield {"type": "status", "text": f"📤 Uploading {len(shots)} opening frames…"}
        for scene, shot in zip(scenes, shots):
            try:
                scene["base_image_url"] = fal_client.upload_bytes(
                    base64.b64decode(shot["first_b64"]),
                    f"shot_{shot['index']}_open.jpg", "image/jpeg")
            except Exception as e:
                yield {"type": "status", "text": f"⚠️ Shot {shot['index']} frame upload failed: {e}"}

        refs = []
        for i, img in enumerate(product_images):
            try:
                refs.append(fal_client.upload_bytes(
                    base64.b64decode(img["b64"]), f"product_photo_{i}.jpg",
                    img.get("media_type", "image/jpeg")))
            except Exception as e:
                yield {"type": "status", "text": f"⚠️ Could not upload product photo {i + 1}: {e}"}
        if not refs and (product or {}).get("image"):
            refs = [product["image"]]
        recipe["product_reference_urls"] = refs
        recipe["product_swap"] = bool(refs)
        recipe["product_source"] = (f"{len(refs)} photo(s) — swapped into every shot"
                                    if refs else "kept from the reference video")

        # Optional subject / environment replacements, same mechanism.
        for slot, images in (("subject", subject_images), ("environment", environment_images)):
            slot_urls = []
            for i, img in enumerate(images or []):
                try:
                    slot_urls.append(fal_client.upload_bytes(
                        base64.b64decode(img["b64"]), f"{slot}_{i}.jpg",
                        img.get("media_type", "image/jpeg")))
                except Exception as e:
                    yield {"type": "status",
                           "text": f"⚠️ Could not upload {slot} photo {i + 1}: {e}"}
            recipe[f"{slot}_reference_urls"] = slot_urls[:4]

        swapped = [SWAP_SLOTS[k]["label"] for k in SWAP_ORDER
                   if recipe.get("product_reference_urls" if k == "product"
                                 else f"{k}_reference_urls")]
        yield {"type": "status", "text": (
            f"✅ Recipe ready — {len(scenes)} shot(s), {recipe['total_duration']}s, "
            + (f"replacing: {', '.join(swapped)}" if swapped
               else "everything kept from the reference"))}
        yield {"type": "done", "recipe": recipe, "transcript": transcript}

    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


# ── Phase 1: analysis ──────────────────────────────────────────────────────

def analyze_stream(video_bytes, product=None, notes=None, mode="recreate",
                   product_images=None, target_duration=None,
                   subject_images=None, environment_images=None):
    """Yields {'type': 'status'|'done', ...}. Final event carries the recipe.

    `mode`:
      "recreate"     — reconstruct the reference shot for shot, starting each
                       shot from its own opening frame. Optionally swaps in a
                       different product. This is what "make a similar video"
                       actually means.
      "same_product" — keep the reference's product but design new shots.
      "my_product"   — borrow only the format for a different product.

    `product_images` is a list of {"b64", "media_type"} for photos the user
    uploaded directly, so a product that is not in the Shopify store yet works.
    """
    product_images = product_images or []
    if mode == "recreate":
        yield from _analyze_recreate(video_bytes, product, notes, product_images,
                                     target_duration, subject_images, environment_images)
        return
    try:
        yield {"type": "status", "text": "🎞️ Reading video…"}
        meta = probe_video(video_bytes)
        if meta["duration"] <= 0:
            yield {"type": "done", "error": "Could not read this video file. Try re-exporting it as MP4 (H.264)."}
            return

        yield {"type": "status", "text": f"🎞️ {meta['duration']}s · {meta['width']}×{meta['height']} · {meta['aspect_ratio']}"}

        frames = extract_frames(video_bytes)
        if not frames:
            yield {"type": "done", "error": "Could not extract frames from this video."}
            return
        yield {"type": "status", "text": f"🖼️ Extracted {len(frames)} frames"}

        yield {"type": "status", "text": "🎙️ Transcribing audio…"}
        transcript = transcribe(video_bytes)
        yield {"type": "status", "text": (
            f"🎙️ Transcript: {len(transcript.split())} words" if transcript
            else "🎙️ No usable audio — analysing visually only"
        )}

        yield {"type": "status", "text": "🧠 Analysing format with Claude…"}

        content = []
        for i, f in enumerate(frames):
            content.append({"type": "text", "text": f"FRAME INDEX {i} (at {f['t']}s):"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": f["media_type"], "data": f["b64"]},
            })

        for i, img in enumerate(product_images, start=1):
            content.append({"type": "text", "text": f"YOUR PRODUCT — PHOTO {i}:"})
            content.append({
                "type": "image",
                "source": {"type": "base64",
                           "media_type": img.get("media_type", "image/jpeg"),
                           "data": img["b64"]},
            })

        mode_block = (MODE_SAME_PRODUCT if mode == "same_product"
                      else MODE_MY_PRODUCT.format(
                          product=_format_product(product, len(product_images))))
        # Default to the reference's own length. Left to free-text notes the
        # model ignores "same length"; as an explicit number it does not have
        # to infer anything, and the result is checked in code afterwards.
        target = float(target_duration) if target_duration else meta["duration"]
        target = max(2.0, round(target, 2))

        content.append({"type": "text", "text": ANALYSIS_PROMPT.format(
            mode_block=mode_block,
            metadata=json.dumps(meta),
            target=(f"{target}s exactly. Your scene durations must add up to this. "
                    + ("This matches the reference." if not target_duration
                       else f"The user asked for this length; the reference is {meta['duration']}s.")),
            transcript=transcript or "(no audio / silent video)",
            notes=notes or "(none)",
        )})

        resp = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")

        recipe = _normalise_recipe(_extract_json(text), meta, target_duration=target)
        recipe["source_duration"] = meta["duration"]
        recipe["target_duration"] = target
        recipe["mode"] = mode

        if recipe.get("duration_corrected_from"):
            yield {"type": "status", "text": (
                f"⏱️ Model wrote {recipe['duration_corrected_from']}s — rescaled scenes "
                f"to the {target}s target")}

        # Anchor the product to real pixels. Text-to-image cannot reproduce a
        # specific physical object, so shots that show it are built by
        # restaging these actual images instead of describing the product.
        #
        # Exactly ONE source wins. Mixing them looked like a free win — more
        # angles for the edit model — but if the uploaded photo and the clip
        # show different products, the model alternates between them and the
        # product visibly changes partway through the video. An uploaded photo
        # is an explicit statement of what the product is, so it beats frames
        # scraped from someone else's ad.
        urls = []

        for i, img in enumerate(product_images):
            try:
                urls.append(fal_client.upload_bytes(
                    base64.b64decode(img["b64"]),
                    f"product_photo_{i}.jpg",
                    img.get("media_type", "image/jpeg")))
            except Exception as e:
                yield {"type": "status", "text": f"⚠️ Could not upload product photo {i + 1}: {e}"}

        if urls:
            recipe["product_source"] = f"{len(urls)} uploaded photo(s)"
        elif product and product.get("image"):
            urls = [product["image"]]
            recipe["product_source"] = "Shopify product photo"
        elif mode == "same_product":
            picked = [i for i in (recipe.get("product_reference_frames") or [])
                      if isinstance(i, int) and 0 <= i < len(frames)]
            if not picked:
                # Middle frames beat the first and last, which are often titles
                # or end cards rather than the product.
                mid = len(frames) // 2
                picked = [mid, min(mid + 2, len(frames) - 1)]
            picked = list(dict.fromkeys(picked))[:4]

            yield {"type": "status", "text": f"📌 Locking product identity from frames {picked}…"}
            for idx in picked:
                try:
                    urls.append(fal_client.upload_bytes(
                        base64.b64decode(frames[idx]["b64"]), f"product_ref_{idx}.jpg", "image/jpeg"))
                except Exception as e:
                    yield {"type": "status", "text": f"⚠️ Could not upload frame {idx}: {e}"}
            recipe["product_source"] = f"{len(urls)} frame(s) from the reference clip"

        recipe["product_reference_urls"] = urls[:8]

        refs = len(recipe.get("product_reference_urls") or [])
        yield {"type": "status", "text": (
            f"✅ Recipe ready — {len(recipe['scenes'])} scenes, {recipe['total_duration']}s, "
            f"{refs} product reference image(s)")}
        yield {"type": "done", "recipe": recipe, "transcript": transcript}

    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


def rewrite_stream(recipe, instructions):
    """Apply a plain-English change to an existing recipe."""
    try:
        yield {"type": "status", "text": "🧠 Applying changes…"}
        resp = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                recipe=json.dumps(recipe, indent=2),
                instructions=instructions,
            )}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        updated = _normalise_recipe(_extract_json(text), {"aspect_ratio": recipe.get("aspect_ratio", "9:16")})

        # The model rewrites the JSON it was shown and has no reason to carry
        # these through, but losing them would silently downgrade a recreation
        # back to invented shots. Re-attach them from the original by index.
        for key in ("mode", "product_reference_urls", "product_swap", "product_source",
                    "product_identity", "product_parts", "soundscape", "source_duration",
                    "subject_reference_urls", "environment_reference_urls",
                    "overlay_mask_url", "overlay_boxes"):
            if key in recipe:
                updated.setdefault(key, recipe[key])
        originals = {s.get("index"): s for s in (recipe.get("scenes") or [])}
        for s in updated["scenes"]:
            src = originals.get(s["index"])
            if src:
                for key in ("base_image_url", "source_start"):
                    if src.get(key) and not s.get(key):
                        s[key] = src[key]
                # How many copies of the product are in frame was counted off the
                # reference frames. The rewrite may legitimately correct it, but
                # if it simply doesn't mention it, falling back to the default of
                # 1 would leave the reference's own product on every subject but
                # the most prominent — so inherit rather than default.
                if "product_count" in src and "product_count" not in s:
                    s["product_count"] = src["product_count"]
            s["product_count"] = _as_count(s.get("product_count"))

        yield {"type": "status", "text": f"✅ Updated — {len(updated['scenes'])} scenes, {updated['total_duration']}s"}
        yield {"type": "done", "recipe": updated}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


# ── Phase 2: generation ────────────────────────────────────────────────────

def estimate_cost(recipe, video_model=DEFAULT_VIDEO_MODEL,
                  avatar_model=avatar_registry.DEFAULT_AVATAR_MODEL,
                  avatar_resolution="480p"):
    """USD estimate from fal's published per-second list prices.

    Bills against the duration we actually REQUEST, not the recipe duration —
    scenes shorter than the model's 4s floor still cost a full 4s, and assembly
    trims the excess away afterwards.
    """
    spec = fal_client.VIDEO_MODELS.get(video_model) or fal_client.VIDEO_MODELS[DEFAULT_VIDEO_MODEL]
    per_second = spec["usd_per_second"]
    try:
        avatar_per_second = avatar_registry.usd_per_second(avatar_model, avatar_resolution)
    except avatar_registry.AvatarError:
        avatar_per_second = avatar_registry.usd_per_second(
            avatar_registry.DEFAULT_AVATAR_MODEL, avatar_resolution)

    # Each active swap slot is its own edit pass per shot that shows it, and
    # subject/environment swaps run on the dearer models.
    swap_cost = sum(
        (fal_client.USD_PER_EDIT if slot == "product" else fal_client.USD_PER_BIG_EDIT)
        for slot in SWAP_ORDER
        if recipe.get("product_reference_urls" if slot == "product"
                      else f"{slot}_reference_urls"))

    # Erasing the reference's burned-in caption is one more edit per shot.
    # Unlike a swap it leads with the cheap rungs — the check that accepts it
    # is free, so a failed cheap attempt costs only itself — and two of them
    # is the realistic worst case before it lands.
    erase_cost = 0.0
    if recipe.get("overlay_mask_url") and recipe.get("overlay_boxes"):
        erase_cost = 2 * min(r["usd"] for r in fal_client.EDIT_LADDER)

    # Generation widens any slot the model cannot fill honestly, which can buy a
    # longer clip. Estimating against the un-widened recipe would quote a price
    # the run then exceeds, so the same fitting is applied here — on copies, so
    # the recipe the user is still reviewing is not altered behind their back.
    scenes = [dict(s) for s in recipe.get("scenes", [])]
    _fit_slots_to_model(scenes, video_model)

    fal_usd = 0.0
    avatar_usd = 0.0
    for s in scenes:
        # Models only offer fixed clip lengths and bill the whole one, even
        # though assembly trims it down — a 2s scene still costs Kling's 5s
        # minimum. Same helper as generation, so the two cannot drift apart.
        billed = float(fal_client._billable_duration(s.get("duration", 4), spec["durations"]))
        if s.get("kind") == "avatar":
            avatar_usd += max(4.0, s.get("duration", 4)) * avatar_per_second
        else:
            # Every b-roll shot needs a first frame. A recreation starts from
            # the reference's own, costing nothing; otherwise it is generated.
            if not s.get("base_image_url"):
                fal_usd += fal_client.USD_PER_IMAGE
            else:
                fal_usd += erase_cost
            fal_usd += swap_cost
            fal_usd += billed * per_second
            if (s.get("voiceover") or "").strip():
                fal_usd += fal_client.USD_PER_TTS_LINE

    avatar_provider = avatar_registry.provider_of(avatar_model)
    if avatar_provider == "fal":
        # Same account, so there is nothing to split out.
        fal_usd += avatar_usd
        avatar_usd = 0.0

    return {
        "usd": round(fal_usd + avatar_usd, 2),
        "fal_usd": round(fal_usd, 2),
        "avatar_usd": round(avatar_usd, 2),
        "avatar_provider": avatar_provider,
    }


def generate_stream(recipe, video_model=DEFAULT_VIDEO_MODEL,
                    avatar_model=avatar_registry.DEFAULT_AVATAR_MODEL,
                    avatar_image_url=None, avatar_voice=None, product_image_url=None,
                    burn_subtitles=True, output_dir=None,
                    narration="auto", narrator_voice=None, ambient=True):
    """Generate every scene on fal.ai, then assemble the final MP4.

    Yields {'type': 'status'|'scene'|'done', ...}. The final event carries
    `filename` (relative to output_dir) on success.
    """
    output_dir = output_dir or video_assembler.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    try:
        scenes = recipe.get("scenes") or []
        if not scenes:
            yield {"type": "done", "error": "Recipe has no scenes."}
            return

        aspect = recipe.get("aspect_ratio", "9:16")
        voice = recipe.get("voice", "Sarah")

        # Do this before anything is billed: a slot the model cannot play out
        # wastes the whole clip, and the fix is free if applied up front.
        was_total = round(sum(s["duration"] for s in scenes), 2)
        widened = _fit_slots_to_model(scenes, video_model)
        for idx, was, now in widened:
            yield {"type": "status", "text": (
                f"⏱️ Scene {idx}: {was}s → {now}s — {video_model} only makes clips of "
                f"{', '.join(fal_client.VIDEO_MODELS[video_model]['durations'])}s, and "
                f"{was}s could only be filled by speeding one up until it looked like "
                "fast-forward")}
        if widened:
            now_total = round(sum(s["duration"] for s in scenes), 2)
            recipe["total_duration"] = now_total
            yield {"type": "status", "text": (
                f"⏱️ The cut is {now_total}s rather than {was_total}s. A shorter one is "
                f"possible only by speeding the action up, which is what made the last "
                f"version look wrong. Seedance 2.5 takes any whole number of seconds, so "
                f"it can hold a tighter cut — at seven times the price per second.")}

        product_refs = list(recipe.get("product_reference_urls") or [])
        if product_image_url and product_image_url not in product_refs:
            product_refs.append(product_image_url)

        # Each slot is opt-in. An empty one means that part of the frame is
        # left exactly as the reference had it.
        swap_refs = {slot: (product_refs if slot == "product"
                            else list(recipe.get(f"{slot}_reference_urls") or []))
                     for slot in SWAP_ORDER}
        active = [SWAP_SLOTS[k]["label"] for k, v in swap_refs.items() if v]
        if active:
            yield {"type": "status", "text": f"🔁 Replacing: {', '.join(active)}"}

        # Only shots built from a text prompt need product references. A
        # recreation starts from the reference's own frame, which already shows
        # the product, so it needs none unless one is being swapped in.
        invented = [s for s in scenes
                    if s.get("kind") == "broll" and s.get("shows_product")
                    and not s.get("base_image_url")]
        if invented and not product_refs:
            yield {"type": "done", "error": (
                f"{len(invented)} scene(s) show the product, but there are no product "
                "reference images. Re-run Analyse, or pick a Shopify product that has a photo — "
                "without one the product would be invented rather than reproduced."
            )}
            return

        avatar_scenes = [s for s in scenes if s.get("kind") == "avatar"]
        if avatar_scenes and not avatar_image_url:
            yield {"type": "done", "error": (
                f"{len(avatar_scenes)} scene(s) need a talking-head actor, but no actor photo was "
                "provided. Upload an actor photo, or ask for those scenes to be changed to b-roll."
            )}
            return

        # Check every provider this recipe will actually touch, before spending
        # anything — a missing HeyGen key should not surface only after five
        # b-roll scenes have already been billed.
        if any(s.get("kind") == "broll" for s in scenes):
            ok, msg = fal_client.check_account()
            if not ok:
                yield {"type": "done", "error": msg}
                return
        if avatar_scenes:
            ok, msg = avatar_registry.check_provider(avatar_registry.provider_of(avatar_model))
            if not ok:
                yield {"type": "done", "error": msg}
                return

        # Assembly runs last, so a broken ffmpeg would only surface after every
        # scene has been billed. Prove the filter graph works first.
        ok, msg = video_assembler.preflight(burn_subtitles)
        if not ok:
            yield {"type": "done", "error": msg}
            return
        yield {"type": "status", "text": f"🔧 {msg}"}

        overlay_mask_url = recipe.get("overlay_mask_url") or ""
        overlay_boxes = recipe.get("overlay_boxes") or []
        if overlay_mask_url and overlay_boxes:
            yield {"type": "status", "text": (
                f"🧽 Erasing the reference's burned-in text from every starting frame "
                f"({len(overlay_boxes)} region(s))")}

        clips = []
        total = len(scenes)
        locked_frame = None      # first rendered product shot, reused to stop drift
        # Same idea, per swap slot, for the recreation path: the first accepted
        # subject / product / location frame becomes a reference for the rest,
        # so identity holds across the cut instead of being re-rolled per shot.
        locked_slots = {}
        attempted = set()        # slots that have had at least one shot at them
        # Anything that quietly fell back to the reference's own footage. These
        # scroll past in the status feed and get lost, so they are collected and
        # repeated at the end where they cannot be missed.
        warnings = []

        # One continuous read sounds human; the same words rendered as separate
        # per-scene clips come out clipped and robotic, because a two-second
        # scene is only about five words spoken in isolation. Fall back to
        # per-scene only when an avatar scene carries its own speech, which a
        # single track laid over the whole timeline would talk over.
        per_scene_voice = bool(avatar_scenes)
        # fal's on_status callback fires from inside a blocking call, so it can't
        # yield — it appends here and we drain the buffer after each step.
        pending = []

        def status(text):
            pending.append(text)

        def drain(label):
            while pending:
                yield {"type": "status", "text": f"   {label} · {pending.pop(0)}"}

        for s in scenes:
            label = f"Scene {s['index']}/{total}"
            yield {"type": "status", "text": f"🎬 {label} ({s['kind']}, {s['duration']}s) — {s.get('shot_description', '')[:80]}"}

            image_url = None      # avatar scenes have none; must not leak from the last loop
            if s["kind"] == "avatar":
                url = avatar_registry.generate(
                    avatar_model,
                    avatar_image_url,
                    s.get("avatar_line") or s.get("voiceover") or "…",
                    voice=avatar_voice or voice,
                    seconds=max(4.0, s["duration"]),
                    scene_prompt=s.get("shot_description"),
                    aspect_ratio=aspect,
                    on_status=status,
                )
                yield from drain(label)
            elif s.get("base_image_url"):
                # Recreation: this shot's own opening frame from the reference
                # is the starting image, so composition, subject, location and
                # framing are inherited rather than reinvented.
                image_url = s["base_image_url"]
                # First, take the reference's own caption off the plate. Doing
                # it before the swaps means no later edit has to preserve
                # lettering that should not be in the finished video at all.
                if overlay_mask_url and overlay_boxes:
                    erased = yield from _erase_overlay(
                        image_url, overlay_mask_url, overlay_boxes, aspect,
                        label, s["index"], status, drain)
                    if erased == image_url:
                        # Warned, not fatal. A caption that survives is the
                        # defect this whole step exists to remove, but it does
                        # not make the finished ad unusable the way the wrong
                        # product does, and stopping here would leave no way
                        # past a caption the models simply cannot repaint.
                        warnings.append(
                            f"scene {s['index']}: the reference's caption is still in the frame")
                    image_url = erased
                wanted = [(slot, refs) for slot, refs in swap_refs.items()
                          if refs and (slot != "product" or s.get("shows_product"))]
                if not wanted:
                    yield {"type": "status", "text": f"   {label} · using the reference's own frame"}
                n_copies = _as_count(s.get("product_count"))
                if n_copies > 1 and any(slot == "product" for slot, _ in wanted):
                    yield {"type": "status", "text": (
                        f"   {label} · {n_copies} copies of the product in frame — "
                        "all of them get replaced")}
                for slot, refs in wanted:
                    before = image_url
                    image_url = yield from _swap_into(
                        image_url, slot, refs, aspect, label, s["index"], recipe,
                        status, drain, count=n_copies,
                        continuity_url=locked_slots.get(slot))
                    # Only a swap that was actually accepted is worth locking:
                    # on failure `_swap_into` hands back the untouched frame,
                    # which still shows the reference's own subject.
                    if image_url != before:
                        if slot not in locked_slots:
                            locked_slots[slot] = image_url
                    else:
                        first_go = slot not in attempted
                        if first_go:
                            # The first shot that needs a slot is the cheap
                            # canary. If no model can do it there, none of the
                            # remaining shots will do it either, and finishing
                            # the run means paying for a whole video that still
                            # advertises the reference's own product. Stop while
                            # the loss is one shot instead of five.
                            yield {"type": "done", "error": (
                                f"Scene {s['index']} is the first shot that needed the "
                                f"{SWAP_SLOTS[slot]['label']} replaced, and none of the four "
                                f"edit models managed it — the frame still shows the "
                                f"reference's own {SWAP_SLOTS[slot]['noun']}.\n\n"
                                "Stopped here rather than generating the rest, because every "
                                "later shot would have failed the same way and the finished "
                                f"video would have been unusable. About "
                                f"${sum(r['usd'] for r in fal_client.EDIT_LADDER):.2f} was "
                                "spent — one shot's worth of edits, not the whole video.\n\n"
                                "The status lines above say what each model got wrong. Usually "
                                f"it is the {SWAP_SLOTS[slot]['noun']} photo: try one shot "
                                "against a plain background, from a similar angle to the "
                                "reference, with nothing else in the picture.")}
                            return
                        # Later shots are different: the earlier ones are paid
                        # for and fine, and a single bad shot is repairable with
                        # 🔧 Fix for the price of one clip.
                        warnings.append(
                            f"scene {s['index']}: kept the reference's own "
                            f"{SWAP_SLOTS[slot]['noun']}")
                    attempted.add(slot)

                url = fal_client.generate_broll(
                    video_model, image_url,
                    _motion_for(s, recipe.get("location_props")),
                    s["duration"], aspect_ratio=aspect, on_status=status)
                yield from drain(label)

            else:
                prompt = s.get("image_prompt") or s.get("shot_description")
                if s.get("shows_product") and product_refs:
                    # Restage the real product rather than describing it — a
                    # text-to-image model asked for "an orange dog life vest"
                    # returns *an* orange vest, not *this* one.
                    #
                    # Each scene is a separate call, so the product drifts —
                    # most visibly in colour — over the course of a video.
                    # Feeding the first accepted frame back in as an extra
                    # reference pins every later scene to what was actually
                    # rendered, not just to the source photos.
                    refs = product_refs if locked_frame is None else [locked_frame] + product_refs
                    image_url = fal_client.edit_image(
                        refs, prompt, aspect_ratio=aspect, on_status=status)
                    yield from drain(label)
                    if locked_frame is None:
                        locked_frame = image_url
                    yield {"type": "status",
                           "text": f"   {label} · first frame built from {len(refs)} product reference(s)"}
                else:
                    image_url = fal_client.generate_image(prompt, aspect_ratio=aspect, on_status=status)
                    yield from drain(label)
                    yield {"type": "status", "text": f"   {label} · first frame ready"}

                # Pass the true scene length — the client rounds up to the
                # model's nearest allowed duration, and assembly trims back down.
                url = fal_client.generate_broll(
                    video_model,
                    image_url,
                    s.get("motion_prompt") or s.get("shot_description") or prompt
                        or HOLD_STILL_MOTION,
                    s["duration"],
                    aspect_ratio=aspect,
                    on_status=status,
                )
                yield from drain(label)

            clip = {"scene": s, "url": url}
            # Keep the exact frame this scene was animated from — swaps and all.
            # Re-running a single scene later then costs one video call instead
            # of repeating the whole edit ladder that produced this frame.
            if s["kind"] != "avatar" and image_url:
                clip["image_url"] = image_url

            # Providers that derive length from the script (HeyGen) can return a
            # clip longer than the recipe's slot. Trimming it would cut the actor
            # off mid-sentence, so the spoken take wins over the planned timing.
            if s["kind"] == "avatar" and avatar_registry.is_fixed_length(avatar_model):
                clip["keep_full_length"] = True

            # Per-scene narration only when an avatar scene forces it (see the
            # continuous track below).
            vo_text = (s.get("voiceover") or "").strip()
            if per_scene_voice and s["kind"] == "broll" and vo_text:
                yield {"type": "status", "text": f"   {label} · voiceover…"}
                clip["voiceover_url"] = fal_client.generate_voiceover(
                    vo_text, voice=voice, on_status=status
                )
                yield from drain(label)

            clips.append(clip)
            yield {"type": "scene", "index": s["index"], "url": url, "kind": s["kind"]}

        global_audio_url = None
        if narration != "off" and not per_scene_voice:
            spoken = [s for s in scenes
                      if s.get("kind") == "broll" and (s.get("voiceover") or "").strip()]
            word_count = sum(len(s["voiceover"].split()) for s in spoken)
            # "auto" skips narration the reference never really had — a couple
            # of stray words become a synthetic-sounding voiceover over footage
            # that was essentially silent.
            if narration == "auto" and word_count < NARRATION_MIN_WORDS:
                if word_count:
                    yield {"type": "status", "text": (
                        f"🔇 Reference is essentially silent ({word_count} words) — "
                        "skipping narration")}
                spoken = []

            if spoken:
                # Each line is recorded on its own and placed at the moment its
                # scene appears. Read as one continuous take they finished well
                # before the picture did and matched nothing on screen, and
                # running separate shouts together with no gaps is most of what
                # made the delivery sound synthetic.
                starts, t = {}, 0.0
                for s in scenes:
                    starts[s["index"]] = t
                    t += float(s.get("duration") or 0)

                lines = []
                for s in spoken:
                    yield {"type": "status", "text": (
                        f"🔊 Recording line for scene {s['index']} "
                        f"at {starts[s['index']]:.1f}s: “{s['voiceover'][:48]}”")}
                    lines.append({
                        "url": fal_client.generate_voiceover(
                            s["voiceover"].strip(), voice=narrator_voice or voice,
                            on_status=status),
                        "start": starts[s["index"]],
                    })
                    yield from drain("Narration")

                track_dir = tempfile.mkdtemp(prefix="vidcloner_vo_")
                try:
                    track = video_assembler.build_narration_track(
                        lines, t, os.path.join(track_dir, "narration.mp3"), track_dir)
                    # Uploaded rather than kept on disk so a later free
                    # Reassemble can still reach it.
                    global_audio_url = fal_client.upload_bytes(
                        open(track, "rb").read(), "narration.mp3", "audio/mpeg")
                    yield {"type": "status", "text": (
                        f"🔊 {len(lines)} line(s) placed on the timeline, "
                        "each at its own scene")}
                except Exception as e:
                    yield {"type": "status", "text": (
                        f"⚠️ Could not place lines on the timeline ({e}) — "
                        "falling back to one continuous read")}
                    global_audio_url = fal_client.generate_voiceover(
                        " ".join(s["voiceover"].strip() for s in spoken),
                        voice=narrator_voice or voice, on_status=status)
                    yield from drain("Narration")
                finally:
                    shutil.rmtree(track_dir, ignore_errors=True)

        yield {"type": "status", "text": "✂️ Assembling final video…"}
        try:
            filename = video_assembler.assemble(
                clips,
                aspect_ratio=aspect,
                burn_subtitles=burn_subtitles,
                output_dir=output_dir,
            )
        except Exception as e:
            # The clips are already paid for — hand them back so assembly can be
            # retried without regenerating anything.
            yield {"type": "done", "error": f"{type(e).__name__}: {e}",
                   "clips": _serialisable(clips), "global_audio_url": global_audio_url}
            return

        # Ambience has to come from the finished cut, so it is a second pass.
        ambient_url = None
        soundscape = (recipe.get("soundscape") or "").strip()
        if ambient and soundscape:
            cut_path = os.path.join(output_dir, filename)
            try:
                yield {"type": "status", "text": f"🌊 Adding sound: {soundscape[:90]}"}
                cut_url = fal_client.upload_bytes(
                    open(cut_path, "rb").read(), "cut.mp4", "video/mp4")
                ambient_url = fal_client.generate_ambient(
                    cut_url, soundscape, recipe.get("total_duration") or 8, on_status=status)
                yield from drain("Sound")
            except Exception as e:
                yield {"type": "status", "text": f"⚠️ Could not generate ambience: {e}"}

        if ambient_url or global_audio_url:
            cut_path = os.path.join(output_dir, filename)
            mixed = video_assembler.add_soundtrack(
                cut_path, output_dir,
                narration_url=global_audio_url, ambient_url=ambient_url)
            try:
                os.remove(cut_path)      # the silent intermediate
            except OSError:
                pass
            filename = mixed

        if warnings:
            # Said again at the end, and carried on the done event, because the
            # one line that reported each of these went past forty status lines
            # ago. A shot that kept the reference's own footage is exactly what
            # the viewer will notice, and 🔧 Fix re-animates just that shot.
            yield {"type": "status", "text": (
                f"⚠️ {len(warnings)} shot(s) did not come out as asked — "
                "use 🔧 Fix on them rather than regenerating the whole video:")}
            for w in warnings:
                yield {"type": "status", "text": f"   ⚠️ {w}"}

        yield {"type": "status", "text": "✅ Done" + (f" (with {len(warnings)} warning(s))"
                                                     if warnings else "")}
        yield {"type": "done", "filename": filename, "scene_urls": [c["url"] for c in clips],
               "clips": _serialisable(clips), "global_audio_url": global_audio_url,
               "ambient_audio_url": ambient_url, "warnings": warnings}

    except fal_client.FalError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


ERASE_MASKED_PROMPT = (
    "The transparent areas of the mask cover lettering that was pasted on top of this "
    "photograph after it was taken.\n\n"
    "Paint into those areas whatever the scene behind them contains — continue the "
    "surrounding surfaces, textures, colours and lighting straight through, so the result "
    "looks like the photograph as it was shot, before any text was added.\n\n"
    "Add nothing: no new objects, no new text, no letters, no decoration. Change nothing "
    "outside the transparent areas."
)

# Deliberately says nothing about watermarks or logos. An earlier wording listed
# them among the things to take off, and fal returned a hard 422
# content_policy_violation on the prompt alone — removing a watermark is a
# copyright-circumvention request to these providers, whatever the actual
# picture is. What we are doing is repainting a caption a video editor added, so
# that is what it now says.
ERASE_PROMPT = (
    "You are performing an inpainting edit on this photograph, not generating a new picture.\n\n"
    "A caption has been laid over it in a video editor. Repaint the area those letters cover "
    "with the scene that belongs behind them, continuing the surrounding surfaces, textures, "
    "colours and lighting straight through, so the picture looks the way it did before the "
    "caption was added.\n\n"
    "Everything else survives untouched: the same subjects, the same poses and positions, the "
    "same background, the same colours, the same lighting, the same camera viewpoint and "
    "framing. Do not add text of any kind, in any language. Return ONE single photograph of "
    "that same scene — never a grid, a collage or a before/after pair."
)


def _overlay_erased(base_url, cand_url, boxes):
    """Did the text actually go, without the rest of the frame being redrawn?

    Free and specific: lettering is dense high-frequency detail, so the edge
    energy inside the marked boxes collapses when it is genuinely painted out
    and barely moves when the model returns the frame unchanged. A vision call
    would cost money to answer a question pixels already answer.

    Returns (gone, ratio) — ratio is the fraction of the original edge energy
    that survived, or (None, None) when it could not be measured.
    """
    try:
        import numpy as np
        import requests as _rq

        mats = []
        for url in (base_url, cand_url):
            raw = (base64.b64decode(url.split(",", 1)[1]) if url.startswith("data:")
                   else _rq.get(url, timeout=120).content)
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                return None, None
            mats.append(arr)
        base, cand = mats
        # The edit models return whatever resolution they like. Rescaling only
        # the candidate would soften it and make an untouched frame look as
        # though the text had been half removed, so both are brought to the
        # smaller of the two and the boxes are scaled with them.
        scale = 1.0
        if cand.shape != base.shape:
            th = min(base.shape[0], cand.shape[0])
            tw = max(1, int(round(base.shape[1] * th / base.shape[0])))
            scale = th / base.shape[0]
            base = cv2.resize(base, (tw, th), interpolation=cv2.INTER_AREA)
            cand = cv2.resize(cand, (tw, th), interpolation=cv2.INTER_AREA)
            boxes = [[max(1, int(round(v * scale))) for v in b] for b in boxes]

        before = after = 0.0
        h, w = base.shape
        for x, y, bw, bh in boxes:
            x2, y2 = min(w, x + bw), min(h, y + bh)
            if x2 <= x or y2 <= y:
                continue
            before += float(np.abs(cv2.Laplacian(base[y:y2, x:x2], cv2.CV_32F, ksize=3)).sum())
            after += float(np.abs(cv2.Laplacian(cand[y:y2, x:x2], cv2.CV_32F, ksize=3)).sum())
        if before <= 0:
            return None, None
        ratio = after / before
        return ratio <= 0.6, ratio
    except Exception:
        return None, None


def _erase_overlay(base_url, mask_url, boxes, aspect, label, scene_index, status, drain):
    """Paint the reference's burned-in caption out of a starting frame.

    Runs before any swap, so every later edit works on a clean plate rather
    than on letters that the swap models would try to preserve.

    Cheap rungs first, which is the opposite of a swap: the check here costs
    nothing, so a $0.04 attempt that fails is only $0.04 wasted. The masked
    model is the backstop — it is the one that physically cannot touch the rest
    of the picture — and it is also the most expensive, so it is not the first
    thing tried.
    """
    # Order is from what these models actually did on this job, not from price.
    # Seedream went first when this was written, on the assumption that a
    # caption is a small local edit and the cheap rung would manage it; handed a
    # real frame it returned an entirely recomposed picture (layout match 0.02
    # against a floor of 0.40). It is last now. Nano Banana leads because a
    # localised repaint is what it is good at, and gpt-image-2 follows as the
    # backstop: with the mask it physically cannot redraw the rest of the shot,
    # which is the exact failure of the rung before it.
    preferred = ["nano-banana/edit", "gpt-image-2", "nano-banana-pro", "seedream"]
    ordered = []
    for key in preferred:
        ordered += [r for r in fal_client.EDIT_LADDER
                    if key in r["id"] and r not in ordered]
    ordered += [r for r in fal_client.EDIT_LADDER if r not in ordered]

    for rung in ordered:
        masked = "gpt-image" in rung["id"]
        # A rung that refuses outright must cost us the next rung, not the whole
        # video. One returned a hard 422 on the prompt and took a paid run down
        # with it at scene 1 of 5 — every clip after it was never generated.
        try:
            candidate = fal_client.edit_image(
                [base_url], ERASE_MASKED_PROMPT if masked else ERASE_PROMPT,
                aspect_ratio=aspect, on_status=status, model_id=rung["id"],
                seed=7000 + scene_index, mask_url=mask_url if masked else None)
        except Exception as e:
            yield from drain(label)      # flush whatever it managed to report
            yield {"type": "status", "text": (
                f"   {label} · {rung['label']} refused the edit "
                f"({type(e).__name__}: {str(e)[:160]}) — escalating")}
            continue
        yield from drain(label)

        gone, ratio = _overlay_erased(base_url, candidate, boxes)
        if gone is None:
            yield {"type": "status", "text": (
                f"   {label} · text removal could not be measured — keeping "
                f"{rung['label']}'s frame")}
            return candidate

        # A model that repaints the caption away but restages the shot has cost
        # us more than the caption did. Same floor as a product swap: erasing a
        # caption disturbs the frame even less than replacing a product, so a
        # threshold that provably passes good swaps cannot be too strict here.
        score = _layout_similarity(base_url, candidate)
        if score is not None and score < 0.45:
            yield {"type": "status", "text": (
                f"   {label} · {rung['label']} redrew the shot while erasing the text "
                f"(layout {score:.2f}) — escalating")}
            continue
        if gone:
            yield {"type": "status", "text": (
                f"   {label} · burned-in text erased by {rung['label']} "
                f"({int((1 - ratio) * 100)}% of it gone)")}
            return candidate
        yield {"type": "status", "text": (
            f"   {label} · {rung['label']} left the text in place "
            f"({int(ratio * 100)}% still there) — escalating")}

    yield {"type": "status", "text": (
        f"   ⚠️ {label} · could not erase the reference's text cleanly, keeping the "
        "original frame")}
    return base_url


def _swap_into(base_url, slot, refs, aspect, label, scene_index, recipe,
               status, drain, count=1, continuity_url=None):
    """Replace one thing in a frame, verifying the result. Yields status events.

    Returns the edited image URL, or the untouched input if no rung of the
    ladder produced something that passed both checks — a shot that keeps the
    original is better than one where the scene has been redrawn.
    """
    spec = SWAP_SLOTS[slot]
    props = recipe.get("location_props") if slot == "environment" else None
    prompts = _slot_prompts(slot, len(refs),
                            parts=recipe.get("product_parts"),
                            brief=recipe.get("environment_brief", ""),
                            props=recipe.get("location_props"),
                            count=count,
                            continuity=bool(continuity_url))
    # Every shot is swapped on its own, so nothing ties one shot's result to
    # the next: a single uploaded photo of a poodle came back as a poodle in
    # four shots and a golden retriever in the fifth. Handing back the frame
    # that was already accepted pins the later shots to what was actually
    # rendered, not just to the source photos.
    edit_refs = refs + ([continuity_url] if continuity_url else [])

    # Changing the location while keeping the subject identical does not work
    # by instruction alone — every model tested moved the dog, turned it, or
    # resubmerged it. Cutting the subject out and handing that as a mask makes
    # it physically impossible to touch, so the location swap leads with it.
    mask_url = None
    if slot == "environment":
        try:
            yield {"type": "status", "text": f"   {label} · isolating the subject…"}
            mask_url = fal_client.segment_subject(base_url, on_status=status)
            yield from drain(label)
        except Exception as e:
            yield {"type": "status", "text": f"   {label} · could not isolate the subject ({e})"}

    ladder = fal_client.edit_ladder(slot)
    if mask_url:
        # Only gpt-image-2 accepts a mask, so it has to go first here.
        masked = [r for r in ladder if "gpt-image" in r["id"]]
        ladder = masked + [r for r in ladder if r not in masked]

    last_note = ""
    for attempt, rung in enumerate(ladder, start=1):
        prompt = prompts[min(attempt - 1, len(prompts) - 1)]
        # Retrying blind repeats the same mistake — a chin-rest pad kept coming
        # back in the original colour. Tell the next attempt what was wrong.
        if last_note:
            prompt += (f"\n\nA previous attempt was REJECTED for this reason: {last_note}\n"
                       "Fix exactly that, while still obeying everything above.")
        use_mask = mask_url if "gpt-image" in rung["id"] else None
        # Pin the recipe's aspect. "auto" lets the model pick, and it returns a
        # differently-cropped frame that no longer matches the rest of the cut.
        # Same reasoning as the erase ladder: one provider refusing has to cost
        # the next rung, not the rest of the video. The ladder exists precisely
        # because these models are unreliable, and a hard error is just another
        # way for a rung to be unreliable.
        try:
            candidate = fal_client.edit_image(
                [base_url] + edit_refs, prompt, aspect_ratio=aspect, on_status=status,
                model_id=rung["id"], seed=attempt * 1000 + scene_index, mask_url=use_mask)
        except Exception as e:
            # `last_note` is deliberately left alone: it is fed to the next
            # attempt as "here is what was wrong with your result", and a
            # transport error says nothing about the picture. Pasting a provider
            # error into the next prompt is also a good way to trip the content
            # checker that may have caused it.
            yield from drain(label)
            yield {"type": "status", "text": (
                f"   {label} · {rung['label']} failed "
                f"({type(e).__name__}: {str(e)[:160]}) — escalating")}
            continue
        yield from drain(label)

        # The edit model will happily compose a brand-new scene out of all the
        # inputs instead of editing the first one — a floating dog in deep
        # water came back standing on a different beach. Cheap structural check
        # first, where the slot allows one.
        score = None
        if spec["layout_min"] is not None:
            score = _layout_similarity(base_url, candidate)
            if score is not None and score < spec["layout_min"]:
                yield {"type": "status", "text": (
                    f"   {label} · {rung['label']} redrew the shot "
                    f"(layout match {score:.2f}) — escalating")}
                continue

        scene_ok, swapped, note = _verify_swap(
            base_url, candidate, refs, slot, removed_props=props,
            masked=bool(use_mask),
            parts=recipe.get("product_parts") if slot == "product" else None,
            count=count if slot == "product" else 1)
        if scene_ok and swapped:
            yield {"type": "status", "text": (
                f"   {label} · {spec['label']} swapped in"
                + (f" (all {count} copies)" if slot == "product" and count > 1 else "")
                + (" (subject mask held)" if use_mask else "")
                + (f", layout {score:.2f}" if score is not None else ""))}
            return candidate

        reason = ("the rest of the shot was redrawn" if not scene_ok
                  else f"the {spec['noun']} was not fully replaced")
        last_note = note or reason
        yield {"type": "status", "text": (
            f"   {label} · {rung['label']} rejected — {reason}"
            + (f": {note}" if note else ""))}

    yield {"type": "status", "text": (
        f"   ⚠️ {label} · could not replace the {spec['noun']} without redrawing the shot, "
        f"so it keeps the original")}
    return base_url


def _serialisable(clips):
    """Everything assembly needs to run again, without the generation cost.

    `image_url` is not used by assembly — it is kept so a single scene can be
    re-animated later from the frame it already had, skipping the edit ladder.
    """
    return [{"scene": c["scene"], "url": c["url"],
             "image_url": c.get("image_url"),
             "voiceover_url": c.get("voiceover_url"),
             "keep_full_length": c.get("keep_full_length", False)} for c in clips]


REVISE_PROMPT = """A video has already been generated from this recipe. The user has watched it and
wants specific things fixed. Work out which scenes have to be re-animated, and how their prompts
should change.

Re-animating a scene costs real money, so choose the SMALLEST set that satisfies the request. A
complaint about one moment means one scene. Only touch a scene the user's words actually point at.

## THE RECIPE
```json
{recipe}
```

## WHAT THE USER WANTS FIXED
{instructions}

## HOW TO WRITE THE REPLACEMENT PROMPTS

The scene's still opening frame is kept — only the motion is generated again. So `motion_prompt` is
the only lever you have over what happens, and it is what went wrong last time. Rewrite it to be
explicit about the thing the user complained about.

The model does not know what is solid, what floats, what bears weight or what is holding still, and
guesses badly when unsaid — paws sink through a floating mat, an animal told it "walks toward the
water" walks ON the water, a floating platform nobody called stationary drifts off into the
distance. So every `motion_prompt` you write must end with a physical-contact sentence: what carries
the subject's weight and stays rigid, which objects are stationary (name them), and where the
waterline sits on a swimming subject. Never use "walks", "steps" or "runs" for movement in water —
use "paddles", "swims", "pushes off", "slides in".

If the user asks for a scene to be longer or shorter, set its `duration` to what they asked for.
If they ask for different words to be said over a scene, set its `voiceover`.

## WHAT TO RETURN

One JSON object in a ```json fenced block, no prose outside it:

{{
  "summary": "one sentence, plain English, on what you are changing",
  "scenes": [
    {{
      "index": <the scene number to re-animate>,
      "why": "what was wrong with it, in the user's terms",
      "motion_prompt": "the full replacement motion prompt, ending with the physical-contact sentence",
      "duration": <new length in seconds — omit entirely to keep the current one>,
      "voiceover": "<new line — omit entirely to keep the current one>"
    }}
  ]
}}

If nothing in the request needs a scene re-animated (for example it only asks for different
narration or a different length), still return the scene entries with the changed fields — leave
`motion_prompt` out and it will not be re-animated, only re-timed."""


def plan_revision(recipe, instructions):
    """Work out which scenes a change request actually touches.

    Returns {"summary": str, "scenes": [{index, motion_prompt?, duration?, ...}]}.
    """
    slim = {
        "title": recipe.get("title"),
        "total_duration": recipe.get("total_duration"),
        "scenes": [{k: v for k, v in s.items()
                    if k in ("index", "duration", "shot_description", "motion_prompt",
                             "voiceover", "shows_product")}
                   for s in (recipe.get("scenes") or [])],
    }
    resp = client.messages.create(
        model=MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": REVISE_PROMPT.format(
            recipe=json.dumps(slim, indent=2), instructions=instructions)}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    plan = _extract_json(text)

    valid = {s["index"] for s in (recipe.get("scenes") or [])}
    cleaned = []
    for item in plan.get("scenes") or []:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if idx not in valid:
            continue
        entry = {"index": idx, "why": str(item.get("why") or "")}
        if str(item.get("motion_prompt") or "").strip():
            entry["motion_prompt"] = item["motion_prompt"].strip()
        if item.get("duration") is not None:
            try:
                entry["duration"] = max(MIN_SCENE, round(float(item["duration"]), 2))
            except (TypeError, ValueError):
                pass
        if str(item.get("voiceover") or "").strip():
            entry["voiceover"] = item["voiceover"].strip()
        cleaned.append(entry)
    return {"summary": str(plan.get("summary") or ""), "scenes": cleaned}


def revise_stream(recipe, clips, instructions, video_model=DEFAULT_VIDEO_MODEL,
                  burn_subtitles=False, output_dir=None, narration="auto",
                  narrator_voice=None, ambient=True, output_filename=None):
    """Re-animate only the scenes a change request touches, then rebuild.

    Every other scene keeps the clip it already has, so a fix to one moment
    costs one video call rather than the whole video again.
    """
    output_dir = output_dir or video_assembler.OUTPUT_DIR
    try:
        by_index = {c["scene"]["index"]: c for c in clips if c.get("scene")}
        if not by_index:
            yield {"type": "done", "error": "This project has no saved scenes to revise."}
            return

        yield {"type": "status", "text": "🧠 Working out what needs redoing…"}
        plan = plan_revision(recipe, instructions)
        if not plan["scenes"]:
            yield {"type": "done", "error": (
                "Could not tell which scene to change from that. Try naming it — "
                "e.g. \"scene 3: the mat should stay still\".")}
            return
        if plan["summary"]:
            yield {"type": "status", "text": f"📝 {plan['summary']}"}

        aspect = recipe.get("aspect_ratio", "9:16")
        scenes_by_index = {s["index"]: s for s in (recipe.get("scenes") or [])}
        pending = []

        def status(text):
            pending.append(text)

        def drain(lbl):
            while pending:
                yield {"type": "status", "text": f"   {lbl} · {pending.pop(0)}"}

        reanimated = 0
        for item in plan["scenes"]:
            idx = item["index"]
            clip = by_index.get(idx)
            scene = scenes_by_index.get(idx)
            if not clip or not scene:
                continue
            label = f"Scene {idx}"

            for field in ("duration", "voiceover"):
                if field in item:
                    value = item[field]
                    if field == "duration":
                        # Same trap as first generation: a slot under the model's
                        # shortest clip divided by MAX_SPEEDUP gets cut instead of
                        # retimed, so the action never finishes on screen.
                        floor = _playable_minimum(video_model)
                        if value < floor:
                            yield {"type": "status", "text": (
                                f"⏱️ {label} · {value}s is too short for {video_model} to "
                                f"play out an action — using {floor}s")}
                            value = floor
                    scene[field] = value
                    clip["scene"][field] = value

            if "motion_prompt" not in item:
                yield {"type": "status", "text": (
                    f"⏱️ {label} · re-timed only, no re-animation needed")}
                continue

            scene["motion_prompt"] = item["motion_prompt"]
            clip["scene"]["motion_prompt"] = item["motion_prompt"]

            frame = clip.get("image_url") or scene.get("base_image_url")
            if not frame:
                yield {"type": "status", "text": (
                    f"⚠️ {label} · no stored frame to re-animate from, skipping")}
                continue

            if item.get("why"):
                yield {"type": "status", "text": f"🎬 {label} — {item['why'][:100]}"}
            yield {"type": "status", "text": (
                f"🎬 {label} · re-animating {scene['duration']}s from its existing frame")}

            clip["url"] = fal_client.generate_broll(
                video_model, frame,
                _ground_motion_prompt(item["motion_prompt"],
                                      recipe.get("location_props")),
                scene["duration"], aspect_ratio=aspect, on_status=status)
            yield from drain(label)
            reanimated += 1
            yield {"type": "scene", "index": idx, "url": clip["url"], "kind": "broll"}

        # Slots may have moved, so narration has to be re-placed even when no
        # scene was re-animated — a line pinned to the old timeline would drift.
        ordered = [by_index[i] for i in sorted(by_index)]
        for c in ordered:
            s = scenes_by_index.get(c["scene"]["index"])
            if s:
                c["scene"] = s
        recipe["total_duration"] = round(
            sum(float(c["scene"].get("duration") or 0) for c in ordered), 2)

        yield {"type": "status", "text": "✂️ Rebuilding the video…"}
        filename = video_assembler.assemble(
            ordered, aspect_ratio=aspect, burn_subtitles=burn_subtitles,
            output_dir=output_dir, filename=output_filename)

        global_audio_url = None
        spoken = [c["scene"] for c in ordered
                  if (c["scene"].get("voiceover") or "").strip()]
        if narration != "off" and spoken:
            word_count = sum(len(s["voiceover"].split()) for s in spoken)
            if narration == "auto" and word_count < NARRATION_MIN_WORDS:
                spoken = []
        if spoken:
            yield {"type": "status", "text": "🔊 Re-recording the lines on the new timing…"}
            starts, t = {}, 0.0
            for c in ordered:
                starts[c["scene"]["index"]] = t
                t += float(c["scene"].get("duration") or 0)
            lines = []
            for s in spoken:
                lines.append({
                    "url": fal_client.generate_voiceover(
                        s["voiceover"].strip(),
                        voice=narrator_voice or recipe.get("voice", "Sarah"),
                        on_status=status),
                    "start": starts[s["index"]],
                })
                yield from drain("Narration")
            track_dir = tempfile.mkdtemp(prefix="vidcloner_vo_")
            try:
                track = video_assembler.build_narration_track(
                    lines, t, os.path.join(track_dir, "narration.mp3"), track_dir)
                global_audio_url = fal_client.upload_bytes(
                    open(track, "rb").read(), "narration.mp3", "audio/mpeg")
            finally:
                shutil.rmtree(track_dir, ignore_errors=True)

        ambient_url = None
        soundscape = (recipe.get("soundscape") or "").strip()
        if ambient and soundscape:
            try:
                yield {"type": "status", "text": "🌊 Re-recording the ambience…"}
                cut_url = fal_client.upload_bytes(
                    open(os.path.join(output_dir, filename), "rb").read(),
                    "cut.mp4", "video/mp4")
                ambient_url = fal_client.generate_ambient(
                    cut_url, soundscape, recipe.get("total_duration") or 8,
                    on_status=status)
                yield from drain("Sound")
            except Exception as e:
                yield {"type": "status", "text": f"⚠️ Could not generate ambience: {e}"}

        if ambient_url or global_audio_url:
            cut_path = os.path.join(output_dir, filename)
            mixed = video_assembler.add_soundtrack(
                cut_path, output_dir, narration_url=global_audio_url,
                ambient_url=ambient_url)
            try:
                os.remove(cut_path)
            except OSError:
                pass
            filename = mixed

        yield {"type": "status", "text": (
            f"✅ Done — {reanimated} scene(s) re-animated, the rest reused")}
        yield {"type": "done", "filename": filename, "recipe": recipe,
               "scene_urls": [c["url"] for c in ordered],
               "clips": _serialisable(ordered),
               "global_audio_url": global_audio_url,
               "ambient_audio_url": ambient_url,
               "reanimated": reanimated}

    except fal_client.FalError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


def reassemble_stream(clips, aspect_ratio="9:16", burn_subtitles=True, output_dir=None,
                      global_audio_url=None, ambient_audio_url=None):
    """Re-run assembly over already-generated clips. Costs nothing."""
    try:
        if not clips:
            yield {"type": "done", "error": "This project has no saved clips to reassemble."}
            return

        ok, msg = video_assembler.preflight(burn_subtitles)
        if not ok:
            yield {"type": "done", "error": msg}
            return

        yield {"type": "status", "text": f"🔧 {msg}"}
        yield {"type": "status", "text": f"✂️ Reassembling {len(clips)} clips…"}
        filename = video_assembler.assemble(
            clips,
            aspect_ratio=aspect_ratio,
            burn_subtitles=burn_subtitles,
            output_dir=output_dir or video_assembler.OUTPUT_DIR,
            global_audio_url=global_audio_url,
            ambient_audio_url=ambient_audio_url,
        )
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename,
               "scene_urls": [c["url"] for c in clips], "clips": clips}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}
