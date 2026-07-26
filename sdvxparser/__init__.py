"""Convenience API for parsing SOUND VOLTEX VOX charts."""
from pathlib import Path

from .classes.chart import ChartInfo
from .parser.vox import VOXParser

__all__ = ["parse_vox"]


def parse_vox(path: str | Path, *, parse_original_vols: bool = False) -> ChartInfo:
    """Parse a VOX chart from ``path`` into :class:`ChartInfo`."""
    with Path(path).open(encoding="utf-8-sig") as file:
        return VOXParser(parse_original_vols=parse_original_vols).parse(file)
