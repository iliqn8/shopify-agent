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
import textwrap
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


def _escape_path(path):
    """ffmpeg filter args need forward slashes and escaped drive colons."""
    return path.replace("\\", "/").replace(":", r"\:")


def _drawtext_filter(text, width, height, tmpdir, tag):
    """Bottom-third caption. Returns a filter string, or '' if unusable.

    Each wrapped line gets its own drawtext so every line is individually
    centred — the bundled ffmpeg's drawtext has no `text_align` option, and a
    single multi-line draw would come out ragged-left inside a centred box.
    """
    font = _font_file()
    if not font or not text.strip():
        return ""

    # Wrap by hand — drawtext has no word wrapping.
    per_line = max(14, int(width / 42))
    lines = textwrap.wrap(text.strip(), width=per_line)[:3]
    if not lines:
        return ""

    size = max(28, int(width / 22))
    line_h = int(size * 1.45)
    bottom = int(height * 0.14)
    top = height - bottom - line_h * len(lines)

    filters = []
    for i, line in enumerate(lines):
        line_path = os.path.join(tmpdir, f"caption_{tag}_{i}.txt")
        with open(line_path, "w", encoding="utf-8") as f:
            f.write(line)
        filters.append(
            f"drawtext=fontfile='{_escape_path(font)}'"
            f":textfile='{_escape_path(line_path)}'"
            f":fontcolor=white:fontsize={size}"
            f":box=1:boxcolor=black@0.55:boxborderw=14"
            f":x=(w-text_w)/2:y={top + i * line_h}"
        )
    return ",".join(filters)


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
    vf = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        f"fps={FPS}",
    ]
    caption_filter = _drawtext_filter(caption, width, height, tmpdir, tag)
    if caption_filter:
        vf.append(caption_filter)

    args = ["-y", "-i", src]
    has_audio = _has_audio(src)

    if voiceover_path:
        args += ["-i", voiceover_path]
        audio_map = ["-map", "1:a:0"]
    elif has_audio:
        audio_map = ["-map", "0:a:0"]
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio_map = ["-map", f"{1}:a:0"]

    args += ["-map", "0:v:0", *audio_map]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += [
        "-vf", ",".join(vf),
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
