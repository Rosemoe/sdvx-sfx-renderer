"""Overlay one-shot sounds assigned to zero-duration FX button notes."""
from __future__ import annotations

import os
from decimal import Decimal

import numpy as np

from vox_parser.classes.chart import ChartInfo

from .audio import overlay_audio


class ShotSFX:
    sample_rate: int

    def _render_fx_shots(
        self,
        chart: ChartInfo,
        output: np.ndarray,
        shots: dict[int, np.ndarray],
        *,
        offset_ms: float,
        volume: float,
    ) -> None:
        """Overlay the assigned shot for each zero-duration special FX note."""
        offset_seconds = Decimal(str(offset_ms)) / Decimal(1000)
        rendered_count = 0
        missing_slots: set[int] = set()

        for note_type, timepoint, fx in sorted(chart.note_data.iter_fxs(), key=lambda item: item[1]):
            if fx.duration != 0 or fx.special <= 0:
                continue

            shot = shots.get(fx.special)
            if shot is None:
                missing_slots.add(fx.special)
                continue

            start = chart._get_elapsed_time(timepoint) + offset_seconds
            start_sample = int(round(float(start) * self.sample_rate))
            overlay_audio(output, shot, start_sample, volume)
            rendered_count += 1

            if os.environ.get("SDVX_FX_DEBUG"):
                print(
                    f"FX shot: {note_type} {chart.timepoint_to_vox(timepoint)} "
                    f"slot={fx.special} start_sample={start_sample}"
                )

        if os.environ.get("SDVX_FX_DEBUG"):
            print(f"Rendered {rendered_count} FX shots; missing slots={sorted(missing_slots)}")
