#!/usr/bin/env python3
"""Convert a PDF-extracted text list into BioLogic-style map rows.

Expected input format (one ID per line):
    555 variable name
    556 another variable

The script prints output like:
    555: ("variable name", "<u4"),
    556: ("another variable", "<u4"),
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

LINE_PATTERN = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")

UINT32_HINTS = (
    "number",
    "count",
    "cycle number",
    "tech num",
    "end buf",
    "iter",
    "nb",
    "index",
    "shot number",
    "pad number",
    "electrode number",
    "half cycle",
    "z cycle",
)

UINT16_HINTS = (
    "i range",
    "range",
)

UINT8_HINTS = (
    "acir/dcir control",
    "regulation",
    "ramp upwards",
    "shuntischanging",
    "modeischanging",
)


def infer_dtype(name: str) -> str:
    """Infer a BioLogic-compatible NumPy dtype from a variable name."""
    lowered = name.lower()

    if any(token in lowered for token in UINT8_HINTS):
        return "<u1"

    if any(token in lowered for token in UINT16_HINTS):
        return "<u2"

    if any(token in lowered for token in UINT32_HINTS):
        return "<u4"

    if any(token in lowered for token in ("/s", "/h", "/a", "/v", "/ohm", "/hz", "/deg", "/cm", "/nm", "/mw", "/w", "/%", "/°c", "/c", "/g", "/l", "/m", "/au", "/rad", "/s.cm", "/min", "/yr")):
        if any(token in lowered for token in ("time", "charge", "discharge", "energy", "capacity", "efficiency", "dq", "q ", "q-", "cycle", "current")):
            return "<f8"
        return "<f4"

    if any(token in lowered for token in ("phase", "conductivity", "resistivity", "transmittance", "fluorescence", "absorbance", "anisotropy", "optical", "wavelength", "temperature", "t ", "p/", "r/", "e/", "i/", "dq", "delta", "custom", "in ", "out ", "spectrum", "ltime", "cd/", "ord/", "ld/", "cce", "cp", "cs", "tan", "loss angle", "thd", "nsd", "nsr", "pmin", "pmax")):
        return "<f4"

    return "<f4"


def parse_pdf_lines(lines: Iterable[str]) -> List[Tuple[int, str]]:
    """Parse lines like `555 variable name` into `(id, name)` tuples."""
    rows: List[Tuple[int, str]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_PATTERN.match(line)
        if not match:
            raise ValueError(f"Could not parse line: {raw_line!r}")
        col_id = int(match.group(1))
        name = match.group(2).strip()
        rows.append((col_id, name))
    return rows


def format_rows(rows: Iterable[Tuple[int, str]], dtype: str | None = None) -> str:
    """Return a formatted Python dict snippet using inferred dtypes when needed."""
    lines = []
    for col_id, name in rows:
        row_dtype = infer_dtype(name) if dtype is None else dtype
        lines.append(f'    {col_id}: ("{name}", "{row_dtype}"),')
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF-extracted column-ID lines into dict rows"
    )
    parser.add_argument("input", help="Path to the extracted text file")
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional override for the NumPy dtype string. If omitted, the script infers it from the variable name.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    lines = input_path.read_text(encoding="utf-8").splitlines()
    rows = parse_pdf_lines(lines)
    print(format_rows(rows, dtype=args.dtype))


if __name__ == "__main__":
    main()
