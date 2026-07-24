# SDVX SFX Renderer

Renders SOUND VOLTEX FX button effects and VOL laser filters over chart audio from a VOX chart. It can also add knob sounds for full laser slams and optional click sounds at BT/FX note starts.

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

## Credits

- [IDA Pro](https://hex-rays.com/ida-pro)
- [silverhawke249/pyKSH-exporter](https://github.com/silverhawke249/pyKSH-exporter)
- [kshootmania/ksm-v2](https://github.com/kshootmania/ksm-v2)
- [kshootmania/ksmaudio](https://github.com/kshootmania/ksmaudio)
- [zacharied/vox2ksh](https://github.com/zacharied/vox2ksh)
- [iDestyKK/2dx_extract](https://github.com/iDestyKK/2dx_extract)
