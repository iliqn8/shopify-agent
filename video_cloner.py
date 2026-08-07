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
import base64
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


def _layout_similarity(url_a, url_b):
    """How much two images share a layout, 0..1. None if either can't be read.

    Compares downscaled greyscale structure, deliberately ignoring colour: a
    genuine product swap changes hue everywhere the product is, but must leave
    the subject, horizon and framing where they were. A recomposed scene moves
    all of that.
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
            arr = cv2.resize(arr, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
            # Edges track layout; raw brightness would punish lighting shifts.
            arr = cv2.Laplacian(arr, cv2.CV_32F, ksize=3)
            arr -= arr.mean()
            norm = float(np.linalg.norm(arr))
            if norm < 1e-6:
                return None
            mats.append(arr / norm)

        return float(np.clip((mats[0] * mats[1]).sum(), 0.0, 1.0))
    except Exception:
        return None


def _verify_swap(base_url, edited_url, product_urls):
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
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(raw).decode()}},
            ]

        content = block(base_url, "IMAGE A — the original frame:")
        content += block(edited_url, "IMAGE B — the edited frame:")
        for i, u in enumerate(product_urls[:3], start=1):
            content += block(u, f"IMAGE C{i} — the replacement product:")
        content.append({"type": "text", "text":
            "B was meant to be A with only the product replaced by the product in the C images.\n\n"
            "Judge the CONTENT of the scene, not the file. Ignore resolution, image size, "
            "compression and small crop differences at the edges — those are expected and "
            "harmless. What matters is whether the same moment is still depicted.\n\n"
            "Answer with JSON only, no prose:\n"
            '{"scene_kept": <true if B shows the SAME scene as A — same subject in the same '
            'pose and position, same background and setting, same other objects in the same '
            'places, same camera viewpoint. false only if the scene was restaged: the subject '
            'moved or changed activity, the setting or background changed, objects moved, or '
            'the camera viewpoint changed>, '
            '"product_swapped": <true if the product in B is clearly the C product rather than '
            'the one in A>, '
            '"note": "<one short sentence on what differs>"}'})

        resp = client.messages.create(model=MODEL, max_tokens=400,
                                      messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        data = _extract_json(text)
        return (bool(data.get("scene_kept")), bool(data.get("product_swapped")),
                str(data.get("note") or ""))
    except Exception:
        return True, True, "(verification unavailable)"


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
  "voice": "<one of: Aria, Roger, Sarah, Laura, Charlie, George, Callum, River, Liam, Charlotte, Alice, Matilda, Will, Jessica, Eric, Chris, Brian, Daniel, Lily, Bill>",
  "scenes": [
    {{
      "index": <shot number, matching the frames above — one entry per shot, no more, no fewer>,
      "shot_description": "what is in this shot, one sentence",
      "motion_prompt": "WHAT MOVES between the opening and closing frame of THIS shot. Compare the three frames and describe the actual change: which way the subject moves, which way the camera moves, what enters or leaves frame. If the frames are nearly identical, say so — e.g. 'the dog stays still, floating gently; almost no camera movement'. Never invent motion that is not evidenced by the frames.",
      "shows_product": <true/false — is the product visible in this shot>,
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

Now analyse the shots and return the JSON."""


RECREATE_SAME_PRODUCT = """## THE PRODUCT

Keep the product exactly as it appears in the reference — this is a reconstruction of the same ad
for the same product. Nothing about it changes."""


RECREATE_SWAP_PRODUCT = """## THE PRODUCT — IT IS BEING SWAPPED

The reference shows one product; the new video must show a DIFFERENT one, attached above as YOUR
PRODUCT photos. Everything else stays identical — same scene, same subject, same action, same
camera. Only the product itself is replaced.

Describe YOUR PRODUCT (from its photos, not the reference's) in `product_identity`.

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


def _swap_prompts(ref_count):
    """Instructions for swapping a product into an existing frame, best first.

    Each pairs with a rung of fal_client.EDIT_LADDER. The first uses Seedream's
    Figure-referencing, which is built for "the thing in Figure 1 becomes the
    thing in Figure 2". The second frames it as inpainting, which is what makes
    Nano Banana actually perform the swap rather than quietly no-op.
    """
    figs = ("Figure 2" if ref_count == 1
            else "Figures 2" + "".join(f", {i}" for i in range(3, ref_count + 2)))
    others = "the image that follows" if ref_count == 1 else f"the {ref_count} images that follow"
    return [
        (
            f"Figure 1 is a photograph. {figs} show a product on its own.\n\n"
            f"Edit Figure 1 so that the product worn by or attached to the subject is replaced "
            f"by the product from {figs}, matching its real colour, shape and markings.\n\n"
            "Change nothing else in Figure 1. Keep the subject in the same pose and the same "
            "position, keep the background, water, horizon and every other object exactly "
            "where they are, keep the lighting and the camera viewpoint, and keep the same "
            f"crop and framing. Do not use the background or setting from {figs} — those are "
            "product photos, and only the product itself should be taken from them."
        ),
        (
            "You are performing an inpainting edit, not generating a new picture.\n\n"
            "IMAGE 1 is the ONLY image whose scene matters. Reproduce IMAGE 1 exactly. Then "
            "paint over ONLY the product region — the item worn by or attached to the subject "
            f"— replacing it with the product shown in {others}.\n\n"
            "Every pixel outside that product region must be identical to IMAGE 1. The "
            "subject stays in the same place in the same pose. The background, water depth, "
            "shoreline, sky and every other object stay exactly as they are. The camera does "
            "not move. The framing and crop do not change. The reference images contribute "
            "the product's appearance and NOTHING ELSE — ignore their backgrounds entirely.\n\n"
            "The replacement must be clearly visible: match the reference product's real "
            "colour, shape and markings, not the colour of the product already in IMAGE 1."
        ),
        (
            f"IMAGE 1 is a photograph to edit. {others.capitalize()} show a replacement "
            "product, on its own, for reference only.\n\n"
            "Return IMAGE 1 with one single change: the product worn by or shown with the "
            "subject is replaced by the reference product, matching its real colour, shape "
            "and markings.\n\n"
            "Everything else in IMAGE 1 must be unchanged — the subject and its exact pose "
            "and position, the background, the water, the horizon, every other object, the "
            "lighting, the shadows, the camera angle, the crop and the framing. Do not move "
            "anything. Do not re-stage or re-imagine the scene. Do not borrow the background "
            "or camera angle from the reference images; they are product photos only."
        ),
    ]


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

def _analyze_recreate(video_bytes, product, notes, product_images, target_duration):
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

        if target_duration and abs(float(target_duration) - meta["duration"]) > 0.5:
            was = _fit_to_target(scenes, float(target_duration))
            if was is not None:
                recipe["duration_corrected_from"] = was
                yield {"type": "status",
                       "text": f"⏱️ Reference is {was}s — scaled shots to your {target_duration}s target"}
        recipe["total_duration"] = round(sum(s["duration"] for s in scenes), 2)
        recipe["target_duration"] = recipe["total_duration"]

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

        yield {"type": "status", "text": (
            f"✅ Recipe ready — {len(scenes)} shot(s), {recipe['total_duration']}s, "
            + ("product swapped in" if refs else "original product kept"))}
        yield {"type": "done", "recipe": recipe, "transcript": transcript}

    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


# ── Phase 1: analysis ──────────────────────────────────────────────────────

def analyze_stream(video_bytes, product=None, notes=None, mode="recreate",
                   product_images=None, target_duration=None):
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
                                     target_duration)
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
                    "product_identity", "source_duration"):
            if key in recipe:
                updated.setdefault(key, recipe[key])
        originals = {s.get("index"): s for s in (recipe.get("scenes") or [])}
        for s in updated["scenes"]:
            src = originals.get(s["index"])
            if src:
                for key in ("base_image_url", "source_start"):
                    if src.get(key) and not s.get(key):
                        s[key] = src[key]

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

    fal_usd = 0.0
    avatar_usd = 0.0
    for s in recipe.get("scenes", []):
        # Models only offer fixed clip lengths and bill the whole one, even
        # though assembly trims it down — a 2s scene still costs Kling's 5s
        # minimum. Same helper as generation, so the two cannot drift apart.
        billed = float(fal_client._billable_duration(s.get("duration", 4), spec["durations"]))
        if s.get("kind") == "avatar":
            avatar_usd += max(4.0, s.get("duration", 4)) * avatar_per_second
        else:
            # Every b-roll shot needs a first frame — either restaged from the
            # product references, or generated from text.
            fal_usd += (fal_client.USD_PER_EDIT if s.get("shows_product")
                        else fal_client.USD_PER_IMAGE)
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
                    burn_subtitles=True, output_dir=None):
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

        product_refs = list(recipe.get("product_reference_urls") or [])
        if product_image_url and product_image_url not in product_refs:
            product_refs.append(product_image_url)

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
        if product_refs:
            yield {"type": "status",
                   "text": f"📌 {len(product_refs)} product reference image(s) locked in"}

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

        clips = []
        total = len(scenes)
        locked_frame = None      # first rendered product shot, reused to stop drift

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
                if product_refs and s.get("shows_product"):
                    image_url = None
                    prompts = _swap_prompts(len(product_refs))
                    for attempt, rung in enumerate(fal_client.EDIT_LADDER, start=1):
                        swap = prompts[min(attempt - 1, len(prompts) - 1)]
                        # Pin the recipe's aspect. "auto" lets the model pick,
                        # and it returns a differently-cropped frame that no
                        # longer matches the rest of the cut.
                        candidate = fal_client.edit_image(
                            [s["base_image_url"]] + product_refs, swap,
                            aspect_ratio=aspect, on_status=status,
                            model_id=rung["id"], seed=attempt * 1000 + s["index"])
                        yield from drain(label)

                        # The edit model will happily compose a brand-new scene
                        # out of all the inputs instead of editing the first
                        # one — a floating dog in deep water came back standing
                        # on a different beach. Cheap structural check first.
                        score = _layout_similarity(s["base_image_url"], candidate)
                        if score is not None and score < COMPOSITION_MIN:
                            yield {"type": "status", "text": (
                                f"   {label} · {rung['label']} redrew the scene "
                                f"(layout match {score:.2f}) — escalating")}
                            continue

                        scene_ok, swapped, note = _verify_swap(
                            s["base_image_url"], candidate, product_refs)
                        if scene_ok and swapped:
                            image_url = candidate
                            yield {"type": "status", "text": (
                                f"   {label} · {rung['label']}: product swapped in, scene intact"
                                + (f" (layout {score:.2f})" if score is not None else ""))}
                            break

                        reason = ("the scene was redrawn" if not scene_ok
                                  else "the product was left unchanged")
                        yield {"type": "status", "text": (
                            f"   {label} · {rung['label']} rejected — {reason}"
                            + (f": {note}" if note else ""))}

                    if image_url is None:
                        image_url = s["base_image_url"]
                        yield {"type": "status", "text": (
                            f"   ⚠️ {label} · could not swap the product without redrawing the "
                            "scene, so this shot keeps the original product")}
                else:
                    image_url = s["base_image_url"]
                    yield {"type": "status", "text": f"   {label} · using the reference's own frame"}

                url = fal_client.generate_broll(
                    video_model, image_url,
                    s.get("motion_prompt") or s.get("shot_description") or "",
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
                    s.get("motion_prompt") or s.get("shot_description") or prompt,
                    s["duration"],
                    aspect_ratio=aspect,
                    on_status=status,
                )
                yield from drain(label)

            clip = {"scene": s, "url": url}

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
        if not per_scene_voice:
            script = " ".join(
                (s.get("voiceover") or "").strip()
                for s in scenes if s.get("kind") == "broll" and (s.get("voiceover") or "").strip()
            ).strip()
            if script:
                yield {"type": "status", "text": "🔊 Recording narration in one take…"}
                global_audio_url = fal_client.generate_voiceover(script, voice=voice, on_status=status)
                yield from drain("Narration")

        yield {"type": "status", "text": "✂️ Assembling final video…"}
        try:
            filename = video_assembler.assemble(
                clips,
                aspect_ratio=aspect,
                burn_subtitles=burn_subtitles,
                output_dir=output_dir,
                global_audio_url=global_audio_url,
            )
        except Exception as e:
            # The clips are already paid for — hand them back so assembly can be
            # retried without regenerating anything.
            yield {"type": "done", "error": f"{type(e).__name__}: {e}",
                   "clips": _serialisable(clips), "global_audio_url": global_audio_url}
            return

        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename, "scene_urls": [c["url"] for c in clips],
               "clips": _serialisable(clips), "global_audio_url": global_audio_url}

    except fal_client.FalError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


def _serialisable(clips):
    """Everything assembly needs to run again, without the generation cost."""
    return [{"scene": c["scene"], "url": c["url"],
             "voiceover_url": c.get("voiceover_url"),
             "keep_full_length": c.get("keep_full_length", False)} for c in clips]


def reassemble_stream(clips, aspect_ratio="9:16", burn_subtitles=True, output_dir=None,
                      global_audio_url=None):
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
        )
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename,
               "scene_urls": [c["url"] for c in clips], "clips": clips}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}
