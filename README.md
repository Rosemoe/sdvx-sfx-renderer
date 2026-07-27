# SDVX SFX Renderer

Render SOUND VOLTEX sound effects over chart audio from a VOX chart.

## Features

- Parse most sections of `.vox` chart files
- Render FX/VOL/AUTOTAB sound effects
- Render click sounds for buttons optionally

## Usage

Install the Python dependencies and make sure `ffmpeg` is available on your `PATH`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Render a chart:

```powershell
python -m sfx_renderer sample\chart.vox sample\audio.s3v -o output\render.wav
```

Add click sounds at every BT/FX note start, including holds:

```powershell
python -m sfx_renderer sample\chart.vox sample\audio.s3v -o output\render.wav --note-hit
```

Use `python -m sfx_renderer --help` to see options for audio offset, knob sounds, and click volume.

### PitchShift Backend

PitchShift uses `librosa` by default; it is included in `requirements.txt`. The optional
`pyrubberband` backend also requires the Rubber Band command-line utility on `PATH` (or
at `RUBBERBAND_EXECUTABLE`). Select it with
`$env:SDVX_PITCH_SHIFT_BACKEND = "rubberband"` before rendering.

## Credits

- [IDA Pro](https://hex-rays.com/ida-pro)
- [silverhawke249/pyKSH-exporter](https://github.com/silverhawke249/pyKSH-exporter)
- [kshootmania/ksm-v2](https://github.com/kshootmania/ksm-v2)
- [kshootmania/ksmaudio](https://github.com/kshootmania/ksmaudio)
- [iDestyKK/2dx_extract](https://github.com/iDestyKK/2dx_extract)
