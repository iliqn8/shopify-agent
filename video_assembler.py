"""ffmpeg assembly for the Video Cloner.

Uses the static binary bundled with imageio-ffmpeg rather than a system install,
so the same code runs on Windows locally and inside the Playwright base image on
Railway without an apt step. Note that imageio-ffmpeg ships ffmpeg only — there
is no ffprobe, so stream detection is done by parsing `ffmpeg -i` stderr.
"""

import os
import re
import shutil
import tempfile
import subprocess

import requests

_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(_DATA_DIR, "generated_videos")

DIMENSIONS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

FPS = 30


class AssemblyError(Exception):
    pass


def ffmpeg_exe():
    """Path to a usable ffmpeg binary."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise AssemblyError(
            "No ffmpeg available. Install it with: pip install imageio-ffmpeg"
        )


def _run(args, timeout=900):
    proc = subprocess.run([ffmpeg_exe()] + args, capture_output=True, timeout=timeout)
    return proc.returncode, proc.stderr.decode("utf-8", "replace")


def _has_audio(path):
    # `ffmpeg -i <file>` with no output exits non-zero but prints stream info.
    _, err = _run(["-i", path], timeout=60)
    return bool(re.search(r"Stream #\d+:\d+.*: Audio:", err))


def _probe_duration(path):
    """Length in seconds, from `ffmpeg -i` stderr. None if unreadable."""
    _, err = _run(["-i", path], timeout=60)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def _font_file():
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _render_caption_png(text, width, height, dest):
    """Draw a bottom-third caption to a transparent PNG. Returns dest, or None.

    Captions are rendered with Pillow and composited via ffmpeg's `overlay`
    rather than drawn with `drawtext`: the static Linux ffmpeg that
    imageio-ffmpeg ships is built without libfreetype, so `drawtext` does not
    exist there at all (it does on the Windows build, which is why this only
    shows up once deployed). `overlay` is a core filter present in every build.
    """
    if not text.strip():
        return None
    font_path = _font_file()
    if not font_path:
        return None

    from PIL import Image, ImageDraw, ImageFont

    size = max(28, int(width / 22))
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Wrap to the measured pixel width rather than a character count, so the
    # box never runs off frame regardless of the font.
    max_text_w = int(width * 0.82)
    words = text.strip().split()
    lines, current = [], ""
    for w in words:
        trial = f"{current} {w}".strip()
        if draw.textlength(trial, font=font) <= max_text_w or not current:
            current = trial
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    lines = lines[:3]

    pad_x, pad_y, gap = 20, 10, 8
    line_h = size + pad_y * 2
    total_h = line_h * len(lines) + gap * (len(lines) - 1)
    top = height - int(height * 0.14) - total_h

    for i, line in enumerate(lines):
        tw = draw.textlength(line, font=font)
        box_w = tw + pad_x * 2
        x0 = (width - box_w) / 2
        y0 = top + i * (line_h + gap)
        draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + line_h],
                               radius=10, fill=(0, 0, 0, 150))
        draw.text((x0 + pad_x, y0 + pad_y), line, font=font, fill=(255, 255, 255, 255))

    img.save(dest)
    return dest


def preflight(burn_subtitles=True):
    """Exercise the real assembly path on a throwaway clip. Returns (ok, message).

    Assembly runs last, so anything broken here is only discovered after every
    scene has already been paid for. This reproduces the same filter graph on a
    generated test source, which costs nothing, before generation starts.
    """
    tmpdir = tempfile.mkdtemp(prefix="vidcloner_preflight_")
    try:
        try:
            exe = ffmpeg_exe()
        except AssemblyError as e:
            return False, str(e)

        src = os.path.join(tmpdir, "src.mp4")
        code, err = _run([
            "-y", "-f", "lavfi", "-i", "testsrc=size=320x568:rate=30:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", src,
        ], timeout=120)
        if code != 0 or not os.path.exists(src):
            return False, f"ffmpeg cannot encode video: {err[-300:]}"

        caption = "Preflight caption check" if burn_subtitles else ""
        try:
            _normalise_segment(
                src, os.path.join(tmpdir, "seg.mp4"),
                duration=1.0, width=320, height=568,
                caption=caption, voiceover_path=None, tmpdir=tmpdir, tag="pf",
            )
        except AssemblyError as e:
            return False, f"Assembly preflight failed: {e}"

        if caption and not _font_file():
            return True, "ffmpeg OK, but no usable font found — captions will be skipped"
        return True, f"ffmpeg OK ({os.path.basename(exe)})"
    except Exception as e:
        return False, f"Assembly preflight error: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _download(url, dest):
    if url.startswith("file://"):
        shutil.copyfile(url[7:], dest)
        return dest
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


# Beyond this, compressing a clip to its slot stops looking like real motion
# and starts looking sped up, so the excess is trimmed instead.
MAX_SPEEDUP = 2.6


def _normalise_segment(src, dest, duration, width, height, caption, voiceover_path, tmpdir, tag,
                       fit="speed"):
    """Fit to length, crop to frame, burn caption, settle on one audio track.

    `duration=None` keeps the source's own length — used for talking-head clips
    whose length is set by the script rather than by us.

    `fit="speed"` retimes a too-long clip into its slot; `fit="trim"` cuts it.
    Speed is the default because video models only emit fixed clip lengths —
    Kling does 5s or 10s — so an 8s shot is generated as 10s and an 2s shot as
    5s. Trimming those shows the first part of an action paced for a longer
    span, which reads as slow motion. Retiming restores the reference's pace.
    """
    caption_png = _render_caption_png(
        caption, width, height, os.path.join(tmpdir, f"caption_{tag}.png")
    ) if caption else None

    # Input indices have to be tracked by hand — the caption image and the
    # audio source both shift them depending on what this segment needs.
    args = ["-y", "-i", src]
    next_idx = 1

    caption_idx = None
    if caption_png:
        args += ["-i", caption_png]
        caption_idx = next_idx
        next_idx += 1

    has_audio = _has_audio(src)
    if voiceover_path:
        args += ["-i", voiceover_path]
        audio_src = f"{next_idx}:a:0"
        next_idx += 1
    elif has_audio:
        audio_src = "0:a:0"
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio_src = f"{next_idx}:a:0"
        next_idx += 1

    # The cut length is always decided here, never by whichever input happens
    # to run out first. `-shortest` used to do the latter, so a scene with a
    # voiceover shorter than its slot was silently trimmed to the narration —
    # a 2.6s scene with a 1.4s line came out 1.4s, and the whole video ended
    # up short of its target.
    out_dur = duration if duration is not None else _probe_duration(src)

    # Retime before anything else, so the caption and the audio see the final
    # timeline rather than the generated one.
    retime = ""
    if fit == "speed" and duration:
        src_dur = _probe_duration(src)
        if src_dur and src_dur > duration + 0.05:
            factor = src_dur / duration
            if factor <= MAX_SPEEDUP:
                retime = f"setpts=PTS/{factor:.4f},"
            # Past the cap the clip is simply cut; -t below handles that.

    chain = (f"[0:v]{retime}scale={width}:{height}:force_original_aspect_ratio=increase,"
             f"crop={width}:{height},fps={FPS}")
    if caption_idx is not None:
        chain += f"[base];[base][{caption_idx}:v]overlay=0:0:format=auto[v]"
    else:
        chain += "[v]"
    # Pad the audio with silence so it can never be the limiting stream.
    chain += f";[{audio_src}]apad,aresample=48000[a]"

    args += ["-filter_complex", chain, "-map", "[v]", "-map", "[a]"]
    if out_dur:
        args += ["-t", f"{out_dur:.3f}"]
    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        dest,
    ]

    code, err = _run(args)
    if code != 0 or not os.path.exists(dest):
        raise AssemblyError(f"ffmpeg failed on segment {tag}: {err[-800:]}")
    return dest


# Integrated loudness targets, LUFS. Generated ambience comes back around 9 dB
# quieter than real phone footage, so it is normalised rather than scaled by a
# guessed factor; under narration it sits well below it.
LOUDNESS_AMBIENT_ALONE = -18
LOUDNESS_AMBIENT_UNDER_SPEECH = -28
LOUDNESS_NARRATION = -16


def _apply_audio(video_path, tmpdir, out_path, narration_url=None, ambient_url=None):
    """Lay narration and/or ambience over the finished cut.

    Narration is one continuous read rather than per-scene clips, which on
    two-second scenes would land as robotic fragments. Ambience is ducked
    slightly under it so speech stays intelligible, but stays loud on its own
    when there is no narration — the sound of the place is most of what makes
    a clip read as filmed rather than rendered.
    """
    if not narration_url and not ambient_url:
        shutil.copyfile(video_path, out_path)
        return out_path

    video_dur = _probe_duration(video_path)
    args = ["-y", "-i", video_path]
    idx, parts, labels = 1, [], []

    if ambient_url:
        # MMAudio hands back a video with the ambience muxed in; ffmpeg reads
        # the audio stream straight out of it.
        args += ["-i", _download(ambient_url, os.path.join(tmpdir, "ambient_source.mp4"))]
        target = LOUDNESS_AMBIENT_UNDER_SPEECH if narration_url else LOUDNESS_AMBIENT_ALONE
        parts.append(f"[{idx}:a]loudnorm=I={target}:TP=-1.5:LRA=11,apad,aresample=48000[amb]")
        labels.append("[amb]")
        idx += 1

    if narration_url:
        args += ["-i", _download(narration_url, os.path.join(tmpdir, "narration.mp3"))]
        parts.append(f"[{idx}:a]loudnorm=I={LOUDNESS_NARRATION}:TP=-1.5:LRA=11,"
                     f"apad,aresample=48000[vo]")
        labels.append("[vo]")
        idx += 1

    if len(labels) == 2:
        # `longest` with apad on both would run forever; -t bounds the output.
        parts.append(f"{labels[0]}{labels[1]}amix=inputs=2:duration=longest:normalize=0[a]")
    else:
        parts.append(f"{labels[0]}anull[a]")

    args += ["-filter_complex", ";".join(parts), "-map", "0:v:0", "-map", "[a]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    if video_dur:
        args += ["-t", f"{video_dur:.3f}"]
    args += ["-movflags", "+faststart", out_path]

    code, err = _run(args)
    if code != 0 or not os.path.exists(out_path):
        raise AssemblyError(f"ffmpeg failed muxing audio: {err[-600:]}")
    return out_path


def add_soundtrack(cut_path, output_dir, narration_url=None, ambient_url=None,
                   filename=None):
    """Mux narration/ambience onto an already-assembled cut, into a new file.

    Ambience has to be generated FROM the finished video, so it cannot be known
    when the cut is built — hence a second pass. It is only a remux, with the
    video stream copied, so it costs a moment rather than a re-encode.
    """
    import uuid
    os.makedirs(output_dir, exist_ok=True)
    filename = filename or f"video_{uuid.uuid4().hex[:12]}.mp4"
    out_path = os.path.join(output_dir, filename)
    tmpdir = tempfile.mkdtemp(prefix="vidcloner_audio_")
    try:
        _apply_audio(cut_path, tmpdir, out_path,
                     narration_url=narration_url, ambient_url=ambient_url)
        return filename
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def assemble(clips, aspect_ratio="9:16", burn_subtitles=True,
             output_dir=None, filename=None, global_audio_url=None,
             ambient_audio_url=None):
    """Stitch generated scene clips into one MP4.

    `clips` is a list of {"scene": <recipe scene dict>, "url": <video url>,
    "voiceover_url": <optional per-scene audio url>}.

    Returns the output filename (basename, inside output_dir).
    """
    if not clips:
        raise AssemblyError("Nothing to assemble")

    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    width, height = DIMENSIONS.get(aspect_ratio, DIMENSIONS["9:16"])

    import uuid
    filename = filename or f"video_{uuid.uuid4().hex[:12]}.mp4"
    out_path = os.path.join(output_dir, filename)

    tmpdir = tempfile.mkdtemp(prefix="vidcloner_")
    try:
        segments = []
        for i, clip in enumerate(clips):
            scene = clip["scene"]
            raw = _download(clip["url"], os.path.join(tmpdir, f"raw_{i}.mp4"))

            vo_path = None
            if clip.get("voiceover_url"):
                vo_path = _download(clip["voiceover_url"], os.path.join(tmpdir, f"vo_{i}.mp3"))

            caption = ""
            if burn_subtitles:
                caption = (scene.get("on_screen_text") or "").strip()
                if not caption:
                    caption = (scene.get("avatar_line") or scene.get("voiceover") or "").strip()

            seg = _normalise_segment(
                raw, os.path.join(tmpdir, f"seg_{i}.mp4"),
                duration=None if clip.get("keep_full_length") else float(scene.get("duration", 4)),
                width=width, height=height,
                caption=caption,
                voiceover_path=vo_path,
                tmpdir=tmpdir, tag=str(i),
                # A talking head is already paced by its own speech; retiming
                # it would chipmunk the delivery.
                fit="trim" if clip.get("keep_full_length") else "speed",
            )
            segments.append(seg)

        # Narration and ambience span the whole video and are muxed on at the
        # end, so build the cut to a scratch file first.
        needs_mux = bool(global_audio_url or ambient_audio_url)
        cut_path = os.path.join(tmpdir, "cut.mp4") if needs_mux else out_path

        if len(segments) == 1:
            shutil.copyfile(segments[0], cut_path)
        else:
            # All segments share identical codec params, so the concat demuxer
            # can stream-copy instead of re-encoding.
            list_path = os.path.join(tmpdir, "concat.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(f"file '{seg.replace(chr(92), '/')}'\n")

            code, err = _run([
                "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                "-c", "copy", "-movflags", "+faststart", cut_path,
            ])
            if code != 0 or not os.path.exists(cut_path):
                # Fall back to a full re-encode if stream copy rejects the mix.
                code, err = _run([
                    "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", cut_path,
                ])
                if code != 0:
                    raise AssemblyError(f"ffmpeg concat failed: {err[-800:]}")

        if needs_mux:
            _apply_audio(cut_path, tmpdir, out_path,
                         narration_url=global_audio_url, ambient_url=ambient_audio_url)

        return filename
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
