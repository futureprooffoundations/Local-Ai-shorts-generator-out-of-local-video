# Local AI Shorts Generator

A fully local pipeline that turns long-form coding livestreams into short-form
vertical clips (TikTok / YouTube Shorts) — transcription, AI-based clip
ranking, cropping, and caption burning, all running on your own machine.
No cloud APIs, no per-clip cost.

## How it works

1. **Transcribe** — [OpenAI Whisper](https://github.com/openai/whisper) runs
   locally to transcribe the source video.
2. **Rank clips** — a local LLM served by [Ollama](https://ollama.com) reads
   the transcript and picks the most engaging segments.
3. **Cut & reframe** — [ffmpeg](https://ffmpeg.org) cuts the selected
   segments, converts them to 9:16 vertical with a blurred/zoomed background
   fill, and crops the code editor and webcam regions into frame.
4. **Burn captions** — the transcript is burned in as captions on the final
   clips.

Everything happens on-device: video, transcript, and model inference never
leave your machine.

## Requirements

**Hardware**
- NVIDIA GPU recommended (CUDA) — this was built and tested on an 8GB VRAM
  card. CPU-only will work but transcription and inference will be much
  slower.

**Software**
- Python **3.12** (this project does *not* currently support 3.13+ — some
  dependencies don't have compatible wheels yet)
- [ffmpeg](https://ffmpeg.org/download.html) installed and available on your
  system PATH
- [Ollama](https://ollama.com/download) installed and running
- A CUDA-capable build of PyTorch (installed separately from
  `requirements.txt` — see below)

## Setup

### 1. Get the right Python version

Check what's installed:
```powershell
py -0
```
You need 3.12 in that list. If it's not there, install it from
[python.org](https://www.python.org/downloads/) (get the 3.12.x release, not
3.13+).

### 2. Clone and enter the project

```powershell
git clone https://github.com/futureprooffoundations/<repo-name>.git
cd <repo-name>
```

### 3. Create a virtual environment (pinned to 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. If PowerShell blocks the
activation script with an execution-policy error, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4. Install dependencies

Install everything **except** torch first:
```powershell
pip install -r requirements.txt
```

Torch's CUDA build isn't hosted on regular PyPI, so it's installed
separately from PyTorch's own index:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> If your GPU/driver doesn't support the `cu128` build, check
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
> for the right index URL for your CUDA version.

Verify CUDA is actually being used:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
This should print `True`. If it prints `False`, torch installed CPU-only —
re-run the install command above and confirm it's pulling from the
`cu128` index, not falling back to the default PyPI wheel.

### 5. Set up Ollama

Pull the model this project uses for clip ranking:
```powershell
ollama pull llama3
```
(Or whichever base model you're using — rename to match your local setup,
e.g. a custom Modelfile tagged `abod:latest`.)

By default Ollama only allows requests from a small set of trusted origins.
If you're calling it from anywhere other than a plain local script, you may
need to allow broader access:
```powershell
$env:OLLAMA_ORIGINS="*"
ollama serve
```

Confirm it's running and reachable:
```powershell
curl http://localhost:11434/api/tags
```

### 6. Confirm ffmpeg is reachable

```powershell
ffmpeg -version
```
If this fails, install ffmpeg and make sure its `bin` folder is added to
your system PATH, then open a new terminal.

## Usage

> ⚠️ The flags below reflect the ones actively in use as of this project's
> latest version. Run `python make_short.py --help` to see the authoritative,
> current list — update this section if anything's drifted.

Basic run:
```powershell
python make_short.py --input path\to\stream_recording.mp4 --output path\to\output_folder
```

Common options:

| Flag | Description |
|---|---|
| `--input` | Path to the source video file |
| `--output` | Folder to write finished clips to |
| `--language English` | Forces Whisper to transcribe as English, preventing it from misdetecting the spoken language partway through |
| `--code-rect "x,y,w,h"` | Crop rectangle for the code editor region, e.g. `"0,170,1920,610"` |
| `--webcam-rect "x,y,w,h"` | Crop rectangle for the webcam region, e.g. `"1390,780,530,300"` |
| `--model` | Which local Ollama model to use for clip ranking |

Crop rectangles are in `x,y,width,height` format, measured in pixels from the
top-left corner of the source video. The defaults above match a 1920×1080
source with the webcam in the bottom-right corner — recalculate these if your
recording layout is different.

## Notes & known limitations

- **Batch uploading:** posting multiple generated clips back-to-back on
  TikTok has been observed to suppress distribution. Spread uploads out.
- **Model size matters for ranking quality:** smaller local models (4B and
  below) tend to struggle with reliably formatted JSON output for clip
  selection. 7B–8B models have been more consistent.
- **No cloud fallback by design:** this project intentionally has no
  cloud-API mode. If Ollama or ffmpeg aren't reachable, it won't run —
  that's expected, not a bug.

## License

## License

MIT — see [LICENSE](LICENSE) for details.

---

Built by [Abdullah Mohammed](https://github.com/futureprooffoundations) —
civil engineer building across STEM: code, AI/ML, and wherever else it goes.