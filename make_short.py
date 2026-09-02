"""
make_short.py

Takes a whisper --word_timestamps JSON transcript + the source video,
asks a LOCAL model (via Ollama) to find the best clip-worthy segment(s),
cuts them, converts to vertical 9:16 (code on top, webcam on bottom),
and burns in captions.

NEW in this version:
  - Per-clip webcam/code rect detection using a local VISION model
    (qwen2.5vl:7b by default, llava:latest as a fallback/alternative),
    since the cam + screen layout is no longer stationary across a stream.
    --code-rect / --webcam-rect are now optional manual overrides / fallback
    if vision detection fails or you pass --no-auto-detect.
  - Output filenames no longer contain numbers or the word "FINAL" —
    just a sanitized version of the clip's hook_title, with letter
    suffixes (_a, _b, ...) added only if two titles collide.

Usage:
    python make_short.py --video "path\\to\\source.mkv" --preview-grid
    python make_short.py --video "path\\to\\source.mkv" --transcript "path\\to\\source.json" --num-clips 3
    python make_short.py --video "..." --transcript "..." --no-auto-detect --code-rect "x,y,w,h" --webcam-rect "x,y,w,h"

Requires (already in your Ai-shorts .venv):
    pip install requests --break-system-packages
    ffmpeg + ffprobe on PATH (you already have this)
    Ollama running locally with a ranking model (abod:latest) AND
    a vision model pulled (qwen2.5vl:7b or llava:latest)

No API key needed - this hits http://localhost:11434 by default.
"""

import argparse
import base64
import json
import string
import subprocess
import sys
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# 1. RANKER
# ---------------------------------------------------------------------------

RANKER_PROMPT = """You are helping select clips from a raw, unedited coding-stream transcript to turn into short-form vertical videos (TikTok/YouTube Shorts).

Below is a transcript with timestamps (in seconds). Most of the transcript is filler, dead air, or fumbling — that's expected, ignore it.

Find the {num_clips} best standalone segments, each between 20 and 60 seconds, that:
- Have a clear hook in the first few seconds (a strong statement, a problem, a reaction)
- Are understandable without earlier context
- Contain a complete moment: a bug found + fixed, a concept explained, a funny/relatable frustration moment, or a clear payoff
- Avoid segments that are mostly "okay... so... wait..." filler with no content

Do not write, complete, or continue any code. Do not continue the transcript.
Your only job is to output a JSON array describing which segments to use.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
[
  {{"start": 123.4, "end": 168.9, "hook_title": "short punchy title, under 8 words", "reason": "one sentence why this clip works"}}
]

Transcript (start_seconds -> end_seconds: text):
{transcript_block}
"""


def load_transcript_segments(transcript_json_path: Path):
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"]


def build_transcript_block(segments) -> str:
    lines = []
    for seg in segments:
        lines.append(f"{seg['start']:.1f} -> {seg['end']:.1f}: {seg['text'].strip()}")
    return "\n".join(lines)


def rank_segments(segments, num_clips: int, model: str = "abod:latest",
                   ollama_url: str = "http://localhost:11434"):
    transcript_block = build_transcript_block(segments)

    max_chars = 12000
    if len(transcript_block) > max_chars:
        half = max_chars // 2
        transcript_block = (
            transcript_block[:half]
            + "\n...[transcript trimmed for length]...\n"
            + transcript_block[-half:]
        )

    prompt = RANKER_PROMPT.format(num_clips=num_clips, transcript_block=transcript_block)

    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3},
        },
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                print("Local model did not return valid JSON. Raw response was:\n", raw, file=sys.stderr)
                raise
        else:
            print("Local model did not return valid JSON. Raw response was:\n", raw, file=sys.stderr)
            raise

    return _normalize_clips(parsed, raw)


def _normalize_clips(parsed, raw_for_error: str):
    """Ollama's format=json guarantees *valid JSON*, not any particular
    shape. Depending on the model, we might get back:
      - a plain list of clip dicts (what we want)
      - a JSON string containing that list (double-encoded)
      - a dict wrapping the list under some key like "clips"/"segments"
      - a single clip dict instead of a list
    Dig through those cases until we find a list of dicts, or give up
    with a clear error.
    """
    for _ in range(5):
        if isinstance(parsed, list):
            if all(isinstance(item, dict) for item in parsed):
                return parsed
            break  # list of strings/other junk -> nothing usable
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
                continue
            except json.JSONDecodeError:
                break
        if isinstance(parsed, dict):
            if "start" in parsed and "end" in parsed:
                return [parsed]  # single clip returned instead of a list
            for value in parsed.values():
                if isinstance(value, list) and all(isinstance(i, dict) for i in value):
                    return value
            break
        break

    print("Local model returned JSON, but not in the expected shape. Raw response was:\n",
          raw_for_error, file=sys.stderr)
    raise ValueError("Could not extract a list of clips from the model's response")


# ---------------------------------------------------------------------------
# 2. CLIPPER
# ---------------------------------------------------------------------------

def cut_clip(video_path: Path, start: float, end: float, out_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(video_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def get_video_dimensions(video_path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"])


def extract_frame(video_path: Path, out_path: Path, timestamp: float = 1.0):
    """Grab a single representative frame from a (short) clip to feed the vision model."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 2b. VISION-BASED RECT DETECTION (per clip — layout is no longer stationary)
# ---------------------------------------------------------------------------

VISION_PROMPT = """You are looking at a single frame from a coding livestream screen recording.

The frame contains two regions somewhere on screen:
1. "code_rect": a code editor, terminal, or browser window showing code/text/logs.
2. "webcam_rect": a smaller rectangular webcam overlay showing a person (face/upper body).

Image dimensions: {width}x{height} pixels. (0,0) is the top-left corner.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{{"code_rect": {{"x": int, "y": int, "w": int, "h": int}}, "webcam_rect": {{"x": int, "y": int, "w": int, "h": int}}}}

x,y is the top-left corner of the region; w,h is its width and height, all in pixels.
Do not include any other keys, comments, or explanation.
"""


def _encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _clamp_rect(rect, width, height):
    x, y, w, h = rect
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return (x, y, w, h)


def _rect_is_sane(rect, width, height):
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return False
    if w < width * 0.05 or h < height * 0.05:
        return False  # implausibly tiny — probably a bad detection
    return True


def detect_rects_with_vision(frame_path: Path, width: int, height: int,
                              model: str = "qwen2.5vl:7b",
                              ollama_url: str = "http://localhost:11434"):
    """Ask a local vision model where the code area and webcam box are in this
    frame. Returns (code_rect, webcam_rect) as (x, y, w, h) tuples, or raises
    on failure so the caller can fall back to manual/fixed rects."""
    img_b64 = _encode_image_b64(frame_path)
    prompt = VISION_PROMPT.format(width=width, height=height)

    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        },
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = json.loads(raw)  # let this raise if the model gave garbage
    cr = data["code_rect"]
    wr = data["webcam_rect"]

    code_rect = (int(cr["x"]), int(cr["y"]), int(cr["w"]), int(cr["h"]))
    webcam_rect = (int(wr["x"]), int(wr["y"]), int(wr["w"]), int(wr["h"]))

    code_rect = _clamp_rect(code_rect, width, height)
    webcam_rect = _clamp_rect(webcam_rect, width, height)

    if not _rect_is_sane(code_rect, width, height) or not _rect_is_sane(webcam_rect, width, height):
        raise ValueError(f"Vision model returned implausible rects: code={code_rect} webcam={webcam_rect}")

    return code_rect, webcam_rect


def resolve_rects_for_clip(raw_clip_path: Path, clip_index: int, auto_detect: bool,
                            vision_model: str, ollama_url: str,
                            fallback_code_rect, fallback_webcam_rect):
    """Per-clip rect resolution: try vision detection first (since cam/screen
    position can change mid-stream), fall back to the fixed rects passed on
    the CLI (or previously-seen good rects) if detection fails or is disabled."""
    if not auto_detect:
        if fallback_code_rect is None or fallback_webcam_rect is None:
            sys.exit("--no-auto-detect requires --code-rect and --webcam-rect to be set.")
        return fallback_code_rect, fallback_webcam_rect

    width, height = get_video_dimensions(raw_clip_path)
    frame_path = raw_clip_path.with_suffix(".detect_frame.png")
    try:
        # grab a frame a couple seconds in so intros/transitions don't confuse it
        extract_frame(raw_clip_path, frame_path, timestamp=1.5)
        code_rect, webcam_rect = detect_rects_with_vision(
            frame_path, width, height, model=vision_model, ollama_url=ollama_url
        )
        print(f"  Vision-detected rects — code={code_rect} webcam={webcam_rect}")
        return code_rect, webcam_rect
    except Exception as e:
        print(f"  Vision detection failed for clip {clip_index} ({e}).", file=sys.stderr)
        if fallback_code_rect is not None and fallback_webcam_rect is not None:
            print("  Falling back to --code-rect/--webcam-rect values.", file=sys.stderr)
            return fallback_code_rect, fallback_webcam_rect
        sys.exit(
            "  No fallback --code-rect/--webcam-rect provided and vision detection failed. "
            "Either fix the vision model/Ollama connection, or pass fallback rects."
        )
    finally:
        frame_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. VERTICAL CONVERTER (true 9:16, code top half / webcam bottom half)
# ---------------------------------------------------------------------------
#
# Final canvas is always exactly 1080x1920 (9:16), split into two
# 1080x960 boxes. Each source rect is scaled to FIT its box while
# preserving its own aspect ratio (no stretching, no distortion).
#
# Because the code rect and webcam rect are much wider/shorter than a
# 1080x960 box, "fit inside the box" leaves empty space top/bottom.
# Instead of solid black there, each box gets a blurred, zoomed-in copy
# of the SAME footage filling the background, with the crisp unstretched
# version layered on top -- no dead black bars, no cropped-off content.

def _fit_with_blur_fill(crop_w, crop_h, crop_x, crop_y, box_w, box_h, in_label, out_label):
    """Build the filter chain for one panel: blurred cover-fill background
    + sharp fit-inside foreground, composited into a box_w x box_h box."""
    bg = f"{out_label}_bg"
    fg = f"{out_label}_fg"
    raw1 = f"{out_label}_raw1"
    raw2 = f"{out_label}_raw2"
    return (
        f"{in_label}crop={crop_w}:{crop_h}:{crop_x}:{crop_y},split=2[{raw1}][{raw2}];"
        f"[{raw1}]scale={box_w}:{box_h}:force_original_aspect_ratio=increase,"
        f"crop={box_w}:{box_h},gblur=sigma=25,setsar=1[{bg}];"
        f"[{raw2}]scale={box_w}:{box_h}:force_original_aspect_ratio=decrease,setsar=1[{fg}];"
        f"[{bg}][{fg}]overlay=(W-w)/2:(H-h)/2,setsar=1[{out_label}];"
    )


def convert_to_vertical_split(clip_path: Path, out_path: Path,
                               code_rect: tuple, webcam_rect: tuple):
    cx, cy, cw, ch = code_rect
    wx, wy, ww, wh = webcam_rect

    box_w, box_h = 1080, 960  # top half + bottom half = 1080x1920 total

    filter_complex = (
        _fit_with_blur_fill(cw, ch, cx, cy, box_w, box_h, "[0:v]", "code")
        + _fit_with_blur_fill(ww, wh, wx, wy, box_w, box_h, "[0:v]", "cam")
        + "[code][cam]vstack=inputs=2,setsar=1[v]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:a", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def dump_preview_grid(video_path: Path, out_path: Path, timestamp: float = 30.0):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vf", "drawgrid=w=100:h=100:t=2:c=red@0.7",
        "-frames:v", "1",
        "-update", "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Preview grid saved to: {out_path.resolve()}")
    print("Each red grid square = 100x100 source pixels.")
    print("Read off the top-left corner (x,y) and size (w,h) of:")
    print("  - the code/terminal area you want to keep")
    print("  - your webcam box")
    print("Then pass them as --code-rect x,y,w,h --webcam-rect x,y,w,h (used as fallback rects)")


# ---------------------------------------------------------------------------
# 4. CAPTIONS
# ---------------------------------------------------------------------------

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Default,Arial,64,&H00FFFFFF,&H00000000,&H00000000,1,4,0,2,60,60,180

[Events]
Format: Layer, Start, End, Style, Text
"""


def seconds_to_ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def build_ass_captions(segments, clip_start: float, clip_end: float, out_path: Path):
    events = []

    for seg in segments:
        if seg["end"] < clip_start or seg["start"] > clip_end:
            continue

        words = seg.get("words")
        if not words:
            start = max(seg["start"], clip_start) - clip_start
            end = min(seg["end"], clip_end) - clip_start
            if end > start:
                events.append((start, end, seg["text"].strip()))
            continue

        chunk = []
        chunk_start = None
        for w in words:
            if w["end"] < clip_start or w["start"] > clip_end:
                continue
            if chunk_start is None:
                chunk_start = w["start"]
            chunk.append(w["word"].strip())
            if len(chunk) >= 5:
                start = max(chunk_start, clip_start) - clip_start
                end = min(w["end"], clip_end) - clip_start
                if end > start:
                    events.append((start, end, " ".join(chunk)))
                chunk = []
                chunk_start = None
        if chunk and chunk_start is not None:
            start = max(chunk_start, clip_start) - clip_start
            end = min(seg["end"], clip_end) - clip_start
            if end > start:
                events.append((start, end, " ".join(chunk)))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER)
        for start, end, text in events:
            text = text.replace("\n", " ")
            f.write(
                f"Dialogue: 0,{seconds_to_ass_time(start)},{seconds_to_ass_time(end)},"
                f"Default,{text}\n"
            )


def burn_captions(video_path: Path, ass_path: Path, out_path: Path):
    ass_str = str(ass_path).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"ass={ass_str}",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 5. FILE NAMING — no digits, no "final"/"finale", dedup via letters
# ---------------------------------------------------------------------------

def sanitize_title(title: str) -> str:
    """Strip everything except letters/spaces/hyphens/underscores, drop any
    digits, and drop the words 'final'/'finale' (case-insensitive) so the
    output filename never contains a number or that word."""
    banned_words = {"final", "finale"}
    cleaned_chars = []
    for c in title:
        if c.isdigit():
            continue
        if c.isalpha() or c in " _-":
            cleaned_chars.append(c)
    cleaned = "".join(cleaned_chars).strip()

    words = [w for w in cleaned.replace("-", " ").replace("_", " ").split(" ") if w]
    words = [w for w in words if w.lower() not in banned_words]

    safe = "_".join(words) if words else "clip"
    return safe


def dedupe_name(base_name: str, used_names: set) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name
    for letter in string.ascii_lowercase:
        candidate = f"{base_name}_{letter}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
    candidate = base_name
    while candidate in used_names:
        candidate += "_x"
    used_names.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to source video (.mkv/.mp4)")
    parser.add_argument("--transcript", required=False, default=None,
                         help="Path to whisper --output_format json file. Not needed with --preview-grid.")
    parser.add_argument("--num-clips", type=int, default=3)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model", default="abod:latest", help="Ollama model tag for ranking")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--preview-grid", action="store_true",
                         help="Just dump one frame with a coordinate grid, then exit.")

    # Rects are now optional: used as a fallback if vision detection is off or fails.
    parser.add_argument("--code-rect", required=False, default=None,
                         help="x,y,w,h of the code/terminal area in SOURCE pixels (fallback only)")
    parser.add_argument("--webcam-rect", required=False, default=None,
                         help="x,y,w,h of the webcam box in SOURCE pixels (fallback only)")

    parser.add_argument("--no-auto-detect", action="store_true",
                         help="Disable per-clip vision-based rect detection; require --code-rect/--webcam-rect.")
    parser.add_argument("--vision-model", default="qwen2.5vl:7b",
                         help="Ollama vision model tag for detecting code/webcam rects per clip "
                              "(e.g. qwen2.5vl:7b or llava:latest)")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.preview_grid:
        dump_preview_grid(video_path, output_dir / "preview_grid.png")
        return

    transcript_path = Path(args.transcript) if args.transcript else None
    if transcript_path is None:
        sys.exit("--transcript is required unless you're using --preview-grid")

    fallback_code_rect = tuple(int(v) for v in args.code_rect.split(",")) if args.code_rect else None
    fallback_webcam_rect = tuple(int(v) for v in args.webcam_rect.split(",")) if args.webcam_rect else None
    auto_detect = not args.no_auto_detect

    try:
        requests.get(args.ollama_url, timeout=3)
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"Can't reach Ollama at {args.ollama_url}. Is it running? "
            f"Try: ollama serve"
        )

    print(f"Loading transcript: {transcript_path}")
    segments = load_transcript_segments(transcript_path)

    print(f"Asking local model ({args.model}) to find {args.num_clips} clip-worthy segments...")
    clips = rank_segments(segments, args.num_clips, model=args.model, ollama_url=args.ollama_url)

    used_names = set()

    for i, clip in enumerate(clips, start=1):
        start, end = clip["start"], clip["end"]
        title = clip.get("hook_title", "clip")
        safe_title = sanitize_title(title)
        safe_name = dedupe_name(safe_title, used_names)

        duration = end - start
        print(f"\n--- Clip: {title} ({start:.1f}s - {end:.1f}s, {duration:.1f}s long) ---")
        print(f"Reason: {clip.get('reason', 'n/a')}")
        if duration > 60:
            print(f"WARNING: this clip is {duration:.1f}s — over 60s clips may not be treated as a Short.")

        raw_clip = output_dir / f"{safe_name}_raw.mp4"
        vertical_clip = output_dir / f"{safe_name}_vertical.mp4"
        ass_path = output_dir / f"{safe_name}.ass"
        final_clip = output_dir / f"{safe_name}.mp4"

        print("Cutting clip...")
        cut_clip(video_path, start, end, raw_clip)

        print("Resolving code/webcam rects for this clip...")
        code_rect, webcam_rect = resolve_rects_for_clip(
            raw_clip, i, auto_detect, args.vision_model, args.ollama_url,
            fallback_code_rect, fallback_webcam_rect,
        )

        print("Converting to vertical 9:16 (code top / webcam bottom)...")
        convert_to_vertical_split(raw_clip, vertical_clip, code_rect, webcam_rect)

        print("Building captions...")
        build_ass_captions(segments, start, end, ass_path)

        print("Burning in captions...")
        burn_captions(vertical_clip, ass_path, final_clip)

        raw_clip.unlink(missing_ok=True)
        vertical_clip.unlink(missing_ok=True)

        print(f"Done: {final_clip}")

    print(f"\nAll clips written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()