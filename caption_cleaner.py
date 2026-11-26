"""Take burned-in on-screen text (auto-captions, titles, watermarks) out of a clip.

Video models like Gemini Omni sometimes caption the spoken words themselves,
whatever the prompt says. This removes them afterwards, on demand:

1. Claude looks at contact sheets (4 frames a second) and says WHICH frames
   carry overlaid text and roughly WHERE — it tells a caption from a label
   printed on the product, which pixels alone cannot.
2. Pixels then pin it down: the exact frames (a caption can start between two
   samples) and a tight box, by comparing with a clean frame of the same clip.
3. Each frame in that span gets the box replaced by the clean frame's pixels,
   with a feathered edge. That is invisible when the background there is still
   (the usual locked-off phone shot). Where something moves through the box,
   that frame is inpainted instead — softer, but no frozen patch.

The source clip is left alone; the result is written as a new file.
Cost: one Claude call (a few cents); no video model is involved.
"""

import base64
import os
import subprocess
import time

import anthropic
import cv2
import numpy as np

import video_assembler

MODEL = "claude-opus-5-5"
USD_PER_M_IN, USD_PER_M_OUT = 4.0, 20.0

SAMPLE_FPS = 4          # frames per second shown to Claude
SMALL_W = 270           # analysis width (grey) — enough to locate a caption
TILE_W = 240            # width of one tile on a contact sheet
COLS, ROWS = 6, 4       # tiles per sheet
DIFF_LEVEL = 35         # grey-level change that counts as "text here"
STILL_RING = 10.0       # mean grey change around the box below which the background is still

claude = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


class CleanError(Exception):
    pass


# ── Reading ────────────────────────────────────────────────────────────────

def _read(path):
    """fps, (w, h), small grey frames, sampled small colour frames [(index, t, img)]."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise CleanError("Could not open the clip.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sh = max(2, round(h * SMALL_W / w))
    step = fps / SAMPLE_FPS
    grey, samples, i, next_sample = [], [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (SMALL_W, sh), interpolation=cv2.INTER_AREA)
        grey.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        if i >= next_sample - 1e-6:
            samples.append((i, i / fps, small))
            next_sample += step
        i += 1
    cap.release()
    if not grey:
        raise CleanError("The clip has no frames.")
    return fps, (w, h), np.stack(grey), samples


def _sheets(samples):
    """JPEG contact sheets; each tile is labelled below the picture, never on it."""
    per = COLS * ROWS
    out = []
    for start in range(0, len(samples), per):
        chunk = samples[start:start + per]
        tiles = []
        for n, (_, t, img) in enumerate(chunk, start):
            th = round(img.shape[0] * TILE_W / img.shape[1])
            pic = cv2.resize(img, (TILE_W, th), interpolation=cv2.INTER_AREA)
            band = np.zeros((22, TILE_W, 3), np.uint8)
            cv2.putText(band, "#%d  %.2fs" % (n, t), (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tile = np.vstack([pic, band])
            tiles.append(cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60)))
        blank = np.zeros_like(tiles[0])
        while len(tiles) % COLS:
            tiles.append(blank)
        rows = [np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)]
        ok, jpg = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(base64.standard_b64encode(jpg.tobytes()).decode())
    return out


# ── Finding the text ───────────────────────────────────────────────────────

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["overlays"],
    "properties": {"overlays": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "first_tile", "last_tile", "x0", "y0", "x1", "y1"],
        "properties": {
            "text": {"type": "string"},
            "first_tile": {"type": "integer"},
            "last_tile": {"type": "integer"},
            "x0": {"type": "number"}, "y0": {"type": "number"},
            "x1": {"type": "number"}, "y1": {"type": "number"},
        }}}},
}

_ASK = """These contact sheets show one AI-generated video, %d frames a second. Every tile is one frame; the label under it is its number and time (the label is not part of the video).

Find every TEXT OVERLAY: words, numbers, captions, subtitles, titles, stickers or watermarks drawn ON TOP of the picture, the way an editing app adds them. AI video models often caption a spoken word for a fraction of a second, so check every tile.

Do NOT report text that physically belongs to the scene: a logo or label printed on a product or its packaging, signs, posters, book covers, screens, clothing prints.

For each overlay give the first and last tile it is visible in, the text, and its box as fractions (0–1) of ONE tile's picture: x0,y0 top-left, x1,y1 bottom-right. Make the box generous rather than tight. If the same text sits in the same place over consecutive tiles, that is one overlay. If there are none, return an empty list."""


def _ask_claude(sheets):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": s}}
               for s in sheets]
    content.append({"type": "text", "text": _ASK % SAMPLE_FPS})
    try:
        r = claude.messages.create(
            model=MODEL, max_tokens=16000,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": content}])
    except anthropic.APIStatusError as e:
        raise CleanError("Claude could not look at the clip (%s): %s" % (e.status_code, e.message))
    except anthropic.APIConnectionError:
        raise CleanError("Could not reach Claude — try again.")
    if r.stop_reason == "refusal":
        raise CleanError("Claude declined to look at this clip.")
    import json
    text = next((b.text for b in r.content if b.type == "text"), "{}")
    cost = (r.usage.input_tokens * USD_PER_M_IN + r.usage.output_tokens * USD_PER_M_OUT) / 1e6
    return json.loads(text).get("overlays", []), cost


def _locate(ov, samples, grey, fps):
    """Exact frame span, tight box (small-res pixels) and the clean reference frame."""
    n = len(samples)
    a = max(0, min(n - 1, int(ov["first_tile"])))
    b = max(a, min(n - 1, int(ov["last_tile"])))
    total, sh, sw = grey.shape
    x0, x1 = sorted((float(ov["x0"]), float(ov["x1"])))
    y0, y1 = sorted((float(ov["y0"]), float(ov["y1"])))
    # Claude's box, widened a little: the pixels decide the real edges.
    bw, bh = (x1 - x0) * sw, (y1 - y0) * sh
    X0 = max(0, int(x0 * sw - bw * 0.15 - 4)); X1 = min(sw, int(x1 * sw + bw * 0.15 + 4))
    Y0 = max(0, int(y0 * sh - bh * 0.15 - 4)); Y1 = min(sh, int(y1 * sh + bh * 0.15 + 4))
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return None

    # The clean frame: the sample just before the text, or just after it.
    ref = samples[a - 1][0] if a > 0 else (samples[b + 1][0] if b + 1 < n else None)
    if ref is None:
        return None          # text on every frame: nothing clean to copy from
    core = [samples[k][0] for k in range(a, b + 1)]

    area = grey[:, Y0:Y1, X0:X1].astype(np.int16)
    clean = area[ref]
    stack = area[core]
    look = np.median(stack, axis=0)
    # Text pixels: away from the clean frame in every core frame, and still while
    # the text is up (a caption does not move; a head or a hand does).
    mask = (np.abs(stack - clean).min(axis=0) > DIFF_LEVEL) & \
           ((stack.max(axis=0) - stack.min(axis=0)) < 40 if len(core) > 1 else True)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) >= 6:
        bx0, bx1 = X0 + xs.min(), X0 + xs.max() + 1
        by0, by1 = Y0 + ys.min(), Y0 + ys.max() + 1
    else:
        mask = None
        bx0, bx1 = int(x0 * sw), int(x1 * sw)
        by0, by1 = int(y0 * sh), int(y1 * sh)
    pad = 5
    bx0, by0 = max(0, bx0 - pad), max(0, by0 - pad)
    bx1, by1 = min(sw, bx1 + pad), min(sh, by1 + pad)

    # Frame span: grow outward while the text pixels look like the text, not like
    # the clean frame. (Without a pixel mask: the sampled span plus half a gap.)
    def shows_text(f):
        if mask is None:
            return False
        px = area[f][mask]
        return float((np.abs(px - look[mask]) < np.abs(px - clean[mask])).mean()) > 0.5
    lo, hi = core[0], core[-1]
    half = int(fps / SAMPLE_FPS / 2)
    if mask is None:
        lo, hi = max(0, lo - half), min(total - 1, hi + half)
    while lo - 1 >= 0 and lo - 1 != ref and shows_text(lo - 1):
        lo -= 1
    while hi + 1 < total and hi + 1 != ref and shows_text(hi + 1):
        hi += 1
    lo, hi = max(0, lo - 1), min(total - 1, hi + 1)     # one frame for a fade
    if lo <= ref <= hi:
        lo, hi = (ref + 1, hi) if ref < core[0] else (lo, ref - 1)
    return {"text": ov.get("text", ""), "lo": lo, "hi": hi, "ref": ref,
            "box": (bx0, by0, bx1, by1)}


# ── Repairing ──────────────────────────────────────────────────────────────

def _still(grey, f, ref, box):
    """Is the background around the box the same as in the clean frame?"""
    x0, y0, x1, y1 = box
    sh, sw = grey.shape[1:]
    r = 6
    X0, Y0, X1, Y1 = max(0, x0 - r), max(0, y0 - r), min(sw, x1 + r), min(sh, y1 + r)
    ring = np.ones((Y1 - Y0, X1 - X0), bool)
    ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    d = np.abs(grey[f, Y0:Y1, X0:X1].astype(int) - grey[ref, Y0:Y1, X0:X1].astype(int))
    return float(d[ring].mean()) < STILL_RING if ring.any() else True


def _feather(h, w, edge):
    m = np.zeros((h, w), np.float32)
    m[edge:h - edge, edge:w - edge] = 1.0
    k = edge * 2 + 1
    return cv2.GaussianBlur(m, (k, k), 0)[..., None]


def _out_name(path):
    stem, ext = os.path.splitext(os.path.basename(path))
    name = stem + "_notext" + ext
    if os.path.exists(os.path.join(video_assembler.OUTPUT_DIR, name)):
        name = "%s_notext_%d%s" % (stem, int(time.time()), ext)
    return name


def clean_stream(path):
    """Yields status events, then {"type": "done", "filename"|"nothing"|"error", ...}."""
    try:
        yield {"type": "status", "text": "🎞️ Reading the clip…"}
        fps, (w, h), grey, samples = _read(path)
        sheets = _sheets(samples)
        yield {"type": "status", "text": "🔍 Claude is looking for on-screen text in %d frames…" % len(samples)}
        overlays, cost = _ask_claude(sheets)
        if not overlays:
            yield {"type": "done", "nothing": True, "cost": cost,
                   "message": "No on-screen text found — the clip is left as it is."}
            return
        fixes = []
        for ov in overlays:
            fx = _locate(ov, samples, grey, fps)
            if fx:
                fixes.append(fx)
                yield {"type": "status", "text": "   „%s“ — %.2f–%.2fs" % (
                    fx["text"], fx["lo"] / fps, (fx["hi"] + 1) / fps)}
            else:
                yield {"type": "status", "text": "   „%s“ — on every frame, no clean frame to copy from; skipped"
                       % ov.get("text", "")}
        if not fixes:
            yield {"type": "done", "error": "The text is on every frame, so there is nothing clean to copy from."}
            return

        scale = w / SMALL_W
        for fx in fixes:
            x0, y0, x1, y1 = fx["box"]
            fx["full"] = (max(0, int(x0 * scale)), max(0, int(y0 * scale)),
                          min(w, int(np.ceil(x1 * scale))), min(h, int(np.ceil(y1 * scale))))
        refs = {fx["ref"] for fx in fixes}
        by_frame = {}
        for fx in fixes:
            for f in range(fx["lo"], fx["hi"] + 1):
                by_frame.setdefault(f, []).append(fx)

        yield {"type": "status", "text": "🧽 Removing it…"}
        # Clean frames first (one may come after the text), then the rewrite.
        cap = cv2.VideoCapture(path)
        ref_img, i = {}, 0
        while len(ref_img) < len(refs):
            ok, frame = cap.read()
            if not ok:
                break
            if i in refs:
                ref_img[i] = frame
            i += 1
        cap.release()

        out_name = _out_name(path)
        out_path = os.path.join(video_assembler.OUTPUT_DIR, out_name)
        cmd = [video_assembler.ffmpeg_exe(), "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h), "-r", "%.6f" % fps, "-i", "-",
               "-i", path, "-map", "0:v", "-map", "1:a?",
               "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        cap = cv2.VideoCapture(path)
        i, patched, inpainted = 0, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for fx in by_frame.get(i, []):
                X0, Y0, X1, Y1 = fx["full"]
                if _still(grey, i, fx["ref"], fx["box"]) and fx["ref"] in ref_img:
                    clean = ref_img[fx["ref"]][Y0:Y1, X0:X1].astype(np.float32)
                    a = _feather(Y1 - Y0, X1 - X0, max(2, min(8, (Y1 - Y0) // 6, (X1 - X0) // 6)))
                    frame[Y0:Y1, X0:X1] = (clean * a + frame[Y0:Y1, X0:X1] * (1 - a)).astype(np.uint8)
                    patched += 1
                else:
                    m = np.zeros(frame.shape[:2], np.uint8)
                    m[Y0:Y1, X0:X1] = 255
                    frame[:] = cv2.inpaint(frame, m, 5, cv2.INPAINT_TELEA)
                    inpainted += 1
            proc.stdin.write(frame.tobytes())
            i += 1
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0 or not os.path.exists(out_path):
            raise CleanError("Writing the clean clip failed: " + err[-300:])

        note = "%d frame(s) replaced from a clean frame" % patched
        if inpainted:
            note += ", %d inpainted (something moved behind the text)" % inpainted
        yield {"type": "status", "text": "   " + note}
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": out_name, "cost": round(cost, 4),
               "removed": [fx["text"] for fx in fixes]}
    except CleanError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": "%s: %s" % (type(e).__name__, e)}
