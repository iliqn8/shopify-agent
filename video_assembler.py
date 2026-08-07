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


def _normalise_segment(src, dest, duration, width, height, caption, voiceover_path, tmpdir, tag):
    """Trim, crop to frame, burn caption, and settle on exactly one audio track.

    `duration=None` keeps the source's own length — used for talking-head clips
    whose length is set by the script rather than by us.
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
        audio_map = f"{next_idx}:a:0"
        next_idx += 1
    elif has_audio:
        audio_map = "0:a:0"
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio_map = f"{next_idx}:a:0"
        next_idx += 1

    chain = (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
             f"crop={width}:{height},fps={FPS}")
    if caption_idx is not None:
        chain += f"[base];[base][{caption_idx}:v]overlay=0:0:format=auto[v]"
    else:
        chain += "[v]"

    args += ["-filter_complex", chain, "-map", "[v]", "-map", audio_map]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-shortest",
        dest,
    ]

    code, err = _run(args)
    if code != 0 or not os.path.exists(dest):
        raise AssemblyError(f"ffmpeg failed on segment {tag}: {err[-800:]}")
    return dest


def assemble(clips, aspect_ratio="9:16", burn_subtitles=True,
             output_dir=None, filename=None):
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
            )
            segments.append(seg)

        if len(segments) == 1:
            shutil.copyfile(segments[0], out_path)
        else:
            # All segments share identical codec params, so the concat demuxer
            # can stream-copy instead of re-encoding.
            list_path = os.path.join(tmpdir, "concat.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(f"file '{seg.replace(chr(92), '/')}'\n")

            code, err = _run([
                "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                "-c", "copy", "-movflags", "+faststart", out_path,
            ])
            if code != 0 or not os.path.exists(out_path):
                # Fall back to a full re-encode if stream copy rejects the mix.
                code, err = _run([
                    "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", out_path,
                ])
                if code != 0:
                    raise AssemblyError(f"ffmpeg concat failed: {err[-800:]}")

        return filename
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
