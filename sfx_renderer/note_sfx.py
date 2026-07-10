"""Overlay sounds for BT and FX button note starts."""
from __future__ import annotations

import os
from decimal import Decimal

import numpy as np

from sdvxparser.classes.chart import ChartInfo

from .audio import overlay_audio


class NoteHitSFX:
    def _render_note_hit_sounds(
        self,
        chart: ChartInfo,
        output: np.ndarray,
        click_audio: np.ndarray,
        *,
        offset_ms: float,
        volume: float,
    ) -> None:
        """Overlay a click at the start of every BT/FX note, including holds."""
        offset_seconds = Decimal(str(offset_ms)) / Decimal(1000)
        hit_count = 0
        for note_type, timepoint, note in sorted(chart.note_data.iter_buttons(), key=lambda item: item[1]):
            start = chart._get_elapsed_time(timepoint) + offset_seconds
            start_sample = int(round(float(start) * self.sample_rate))
            overlay_audio(output, click_audio, start_sample, volume)
            hit_count += 1

            if os.environ.get("SDVX_FX_DEBUG"):
                print(f"Note hit: {note_type} {chart.timepoint_to_vox(timepoint)} start_sample={start_sample}")

        if os.environ.get("SDVX_FX_DEBUG"):
            print(f"Rendered {hit_count} BT/FX note-start clicks")
