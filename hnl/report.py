"""Markdown table rendering without optional dependencies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


def _fmt(value: object, fmt: str) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if math.isinf(value):
            return "∞"
        return format(value, fmt)
    return str(value).replace("|", "\\|")


def md_table(
    df: pd.DataFrame,
    columns: Sequence[str],
    headers: Optional[Sequence[str]] = None,
    fmt: str = ".3f",
    formats: Optional[Dict[str, str]] = None,
) -> str:
    """Render selected columns as a GitHub-flavoured Markdown table."""
    headers = list(headers) if headers is not None else list(columns)
    formats = formats or {}
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in df.iterrows():
        cells = [_fmt(row[c], formats.get(c, fmt)) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")
    return path
