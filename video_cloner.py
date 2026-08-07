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

# Kling 2.5 Turbo Standard is ~6x cheaper per second than Seedance 2.0 Fast for
# comparable image-to-video work, so it is the sane default for ad iteration.
DEFAULT_VIDEO_MODEL = "kling-2.5-standard"

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
rhythm — and apply it to a DIFFERENT product, described below. You are NOT copying the reference's
subject, only how it is constructed.

## THE PRODUCT THIS NEW VIDEO IS FOR
{product}

Shots that show the product will be rendered from the REAL product photos, so write `image_prompt`
as a staging instruction ("Place this exact product in ...") rather than re-describing the product."""


ANALYSIS_PROMPT = """You are a short-form video ad director. You are given frames from a REFERENCE
video ad (in chronological order, each labelled with its timestamp and a FRAME INDEX), its
transcript, and technical metadata.

{mode_block}

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
  "format_analysis": "2-4 sentences: what format is this, why does it work, what is the hook mechanism",
  "product_identity": "one precise sentence naming what the product physically is, including its distinguishing visual features (colour, shape, markings). Read it off the frames.",
  "product_reference_frames": [<frame indices, 2 to 4 of them, whose frames show the PRODUCT most clearly and unobstructed — prefer close, well-lit, front-or-three-quarter views. These exact frames get fed to the image model as ground truth for what the product looks like.>],
  "total_duration": <number, seconds — match the reference within ~20%, UNLESS the user's instructions specify a length, which always wins>,
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
   uniform 5s scenes. If the user asked for a shorter total than the reference, keep the reference's
   CUT RHYTHM and drop scenes to fit — do not stretch every scene out to pad the runtime.
2. THE FIRST SCENE IS THE HOOK. Whatever mechanism the reference uses in its first 3 seconds
   (question, bold claim, visual pattern-break, problem shown), use the SAME mechanism with the new
   product's angle.
3. "avatar" IS ONLY FOR SHOTS WHERE A PERSON TALKS TO CAMERA. Everything else is "broll". If the
   reference is pure b-roll with voiceover, produce zero avatar scenes. If it is a UGC creator
   talking with cutaways, mirror that mix exactly.
4. MINIMUM SCENE DURATION IS 4 SECONDS for generation purposes. If the reference cuts faster,
   still write the true short duration — assembly trims the generated clip down to it. Never write
   a duration below 0.8s.
5. IMAGE PROMPTS MUST NOT NAME REAL BRANDS, celebrities, or copyrighted characters, and must not
   describe the reference video's specific actors. Describe generic people by role and appearance.
   This does NOT apply to the product itself — keeping the product identical is the whole point.
6. VOICEOVER + AVATAR LINES TOGETHER MUST BE SPEAKABLE IN THE SCENE'S DURATION. Roughly 2.5 words
   per second. A 3-second scene gets ~7 words, not a sentence.
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


def _format_product(product):
    if not product:
        return "(no product selected — write the recipe generically for a direct-response ecommerce product)"
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


def _normalise_recipe(recipe, meta):
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
            s["duration"] = max(0.8, round(float(s.get("duration", 4)), 2))
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
    recipe["total_duration"] = round(sum(s["duration"] for s in clean), 2)
    recipe.setdefault("product_identity", "")
    recipe.setdefault("product_reference_urls", [])
    return recipe


# ── Phase 1: analysis ──────────────────────────────────────────────────────

def analyze_stream(video_bytes, product=None, notes=None, mode="same_product"):
    """Yields {'type': 'status'|'done', ...}. Final event carries the recipe.

    `mode` is "same_product" (rebuild an ad for the product that appears in the
    reference clip) or "my_product" (borrow the format for a Shopify product).
    """
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

        mode_block = (MODE_SAME_PRODUCT if mode == "same_product"
                      else MODE_MY_PRODUCT.format(product=_format_product(product)))
        content.append({"type": "text", "text": ANALYSIS_PROMPT.format(
            mode_block=mode_block,
            metadata=json.dumps(meta),
            transcript=transcript or "(no audio / silent video)",
            notes=notes or "(none)",
        )})

        resp = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")

        recipe = _normalise_recipe(_extract_json(text), meta)
        recipe["source_duration"] = meta["duration"]
        recipe["mode"] = mode

        # Anchor the product to real pixels. Text-to-image cannot reproduce a
        # specific physical object, so the shots that show it are built by
        # restaging these actual images instead of describing the product.
        if mode == "same_product":
            picked = [i for i in (recipe.get("product_reference_frames") or [])
                      if isinstance(i, int) and 0 <= i < len(frames)]
            if not picked:
                # Middle frames beat the first and last, which are often titles
                # or end cards rather than the product.
                mid = len(frames) // 2
                picked = [mid, min(mid + 2, len(frames) - 1)]
            picked = list(dict.fromkeys(picked))[:4]

            yield {"type": "status", "text": f"📌 Locking product identity from frames {picked}…"}
            urls = []
            for idx in picked:
                try:
                    urls.append(fal_client.upload_bytes(
                        base64.b64decode(frames[idx]["b64"]), f"product_ref_{idx}.jpg", "image/jpeg"))
                except Exception as e:
                    yield {"type": "status", "text": f"⚠️ Could not upload frame {idx}: {e}"}
            recipe["product_reference_urls"] = urls
        elif product and product.get("image"):
            recipe["product_reference_urls"] = [product["image"]]

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
        billed = max(4.0, s.get("duration", 4))
        if s.get("kind") == "avatar":
            avatar_usd += billed * avatar_per_second
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

        product_scenes = [s for s in scenes if s.get("kind") == "broll" and s.get("shows_product")]
        if product_scenes and not product_refs:
            yield {"type": "done", "error": (
                f"{len(product_scenes)} scene(s) show the product, but there are no product "
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
            else:
                prompt = s.get("image_prompt") or s.get("shot_description")
                if s.get("shows_product") and product_refs:
                    # Restage the real product rather than describing it — a
                    # text-to-image model asked for "an orange dog life vest"
                    # returns *an* orange vest, not *this* one.
                    image_url = fal_client.edit_image(
                        product_refs, prompt, aspect_ratio=aspect, on_status=status)
                    yield from drain(label)
                    yield {"type": "status",
                           "text": f"   {label} · first frame built from {len(product_refs)} product reference(s)"}
                else:
                    image_url = fal_client.generate_image(prompt, aspect_ratio=aspect, on_status=status)
                    yield from drain(label)
                    yield {"type": "status", "text": f"   {label} · first frame ready"}

                url = fal_client.generate_broll(
                    video_model,
                    image_url,
                    s.get("motion_prompt") or s.get("shot_description") or prompt,
                    max(4.0, s["duration"]),
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

            # Voiceover is generated per scene, not as one continuous track —
            # a single blob would drift out of sync as soon as an avatar scene
            # (which carries its own speech) sits in the middle of the timeline.
            vo_text = (s.get("voiceover") or "").strip()
            if s["kind"] == "broll" and vo_text:
                yield {"type": "status", "text": f"   {label} · voiceover…"}
                clip["voiceover_url"] = fal_client.generate_voiceover(
                    vo_text, voice=voice, on_status=status
                )
                yield from drain(label)

            clips.append(clip)
            yield {"type": "scene", "index": s["index"], "url": url, "kind": s["kind"]}

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
                   "clips": _serialisable(clips)}
            return

        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename, "scene_urls": [c["url"] for c in clips],
               "clips": _serialisable(clips)}

    except fal_client.FalError as e:
        yield {"type": "done", "error": str(e)}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}


def _serialisable(clips):
    """Everything assembly needs to run again, without the generation cost."""
    return [{"scene": c["scene"], "url": c["url"],
             "voiceover_url": c.get("voiceover_url"),
             "keep_full_length": c.get("keep_full_length", False)} for c in clips]


def reassemble_stream(clips, aspect_ratio="9:16", burn_subtitles=True, output_dir=None):
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
        )
        yield {"type": "status", "text": "✅ Done"}
        yield {"type": "done", "filename": filename,
               "scene_urls": [c["url"] for c in clips], "clips": clips}
    except Exception as e:
        yield {"type": "done", "error": f"{type(e).__name__}: {e}"}
