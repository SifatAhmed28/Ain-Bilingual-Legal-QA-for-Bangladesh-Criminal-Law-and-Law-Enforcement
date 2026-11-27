"""Extract CrPC Schedule II (tabular statement of offences) from the official PDF.

The official schedule is a scanned table.  Its text layer keeps cell fragments
as positioned lines and blocks, but many pages merge several adjacent cells
into one block, and a few pages lost their ruling lines entirely.  The
extractor therefore:

1. finds row starts from text-block tops (a block is a cell or a run of cells
   in one table row, so its top marks a row boundary);
2. buckets individual text *lines* into the eight schedule columns, which is
   robust to merged cell blocks;
3. snaps the drawn vertical rules to the actual text-column starts, because
   on some scanned pages the rule layer is offset from the text;
4. carries a section number forward to the following variant rows, as the
   printed schedule merges multi-variant offences under one number.

Every emitted row keeps its page, raw cells and parser status.  Rows are
candidates for review, never legally verified facts.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

try:
    import pymupdf
except ImportError:  # older PyMuPDF
    import fitz as pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_URL = "https://bdlaws.minlaw.gov.bd/upload/act/2025-08-17-15-16-07-75___170825-Schedule.pdf"
AMENDMENT_URL = "https://bdlaws.minlaw.gov.bd/act-print-1640/section-58332.html"
PDF_PATH = ROOT / "data" / "raw" / "bdlaws" / "crpc_schedule_ii_2025.pdf"
OUT_JSON = ROOT / "data" / "clean" / "procedure.json"
OUT_CSV = ROOT / "data" / "clean" / "procedure_auto_candidates.csv"

DEFAULT_X = (160.0, 220.0, 264.0, 306.0, 345.0, 400.0, 456.0)
PAGE_LEFT, PAGE_RIGHT = 118.0, 492.0
DITTO = {"ditto", "do", "do."}
HEADING = re.compile(r"^(?:CHAPTER\b|PART\b|Of\s|Offences\s)")


def clean(text: str) -> str:
    text = str(text or "").replace("\u00a0", " ").replace("\uf0ac", "")
    text = re.sub(r"\s+", " ", text).strip()
    # Rejoin words broken across PDF lines ("Compoun- dable").
    text = re.sub(r"(?<=[a-z])- (?=[a-z])", "", text)
    # Footnote markers: "1[...]" -> "[...]"
    text = re.sub(r"^\d+\s*\[", "[", text)
    text = re.sub(r"\s+\d+\s*\[", " [", text)
    return text.strip()


def download_pdf() -> None:
    import requests

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 100_000:
        return
    response = requests.get(PDF_URL, timeout=180)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError("Official schedule URL did not return a PDF")
    PDF_PATH.write_bytes(response.content)


def schedule_page(page) -> bool:
    text = page.get_text("text").casefold()
    return "offence" in text and "bailable" in text and ("punish" in text or "penal code" in text)


def page_rules(page) -> tuple[float, float] | None:
    ys = set()
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "re":
                continue
            rect = item[1]
            width, height = rect.x1 - rect.x0, rect.y1 - rect.y0
            if height <= 1.5 and width > 1.5:
                ys.add(round(float(rect.y0), 1))
    ys = sorted(ys)
    return (ys[0], ys[-1]) if len(ys) >= 2 else None


def rule_columns(page) -> list[float]:
    """Seven internal column boundaries from the drawn vertical rules."""
    lines = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "re":
                continue
            rect = item[1]
            width, height = rect.x1 - rect.x0, rect.y1 - rect.y0
            if width <= 1.5 and height > 1.5:
                lines.append((round(float(rect.x0), 1), float(height)))
    merged: list[tuple[float, float]] = []
    for x, span in sorted(lines):
        if merged and x - merged[-1][0] <= 8:
            if span > merged[-1][1]:
                merged[-1] = (x, span)
        else:
            merged.append((x, span))
    xs = [x for x, _ in merged]
    return xs[:7] if len(xs) >= 6 else list(DEFAULT_X)


def text_lines(page, top: float, bottom: float):
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            if y1 <= top or y0 >= bottom:
                continue
            text = clean("".join(span["text"] for span in line["spans"]))
            if text:
                lines.append((x0, y0, x1, y1, text))
    return lines


def cell_fragments(page, top: float, bottom: float, boundaries: list[float]):
    """Return (column, y0, y1, text) fragments for one table page.

    A block inside one column is one cell.  A block spanning columns is split
    into its lines so every line lands in its own column.
    """
    fragments = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        lines = []
        for line in block["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            if y1 <= top or y0 >= bottom:
                continue
            text = clean("".join(span["text"] for span in line["spans"]))
            if text:
                lines.append((x0, y0, x1, y1, text))
        if not lines:
            continue
        columns = {column_of(l[0], l[2], boundaries) for l in lines}
        if len(columns) == 1:
            text = clean(" ".join(l[4] for l in lines))
            if HEADING.match(text):
                continue
            fragments.append((columns.pop(), min(l[1] for l in lines), max(l[3] for l in lines), text))
        else:
            for l in lines:
                if HEADING.match(l[4]):
                    continue
                fragments.append((column_of(l[0], l[2], boundaries), l[1], l[3], l[4]))
    return fragments


def assign_cells(fragments: list[tuple], column: int, rows: list[tuple[float, float]]) -> list[str]:
    """Assign fragments of one column to rows, in printed row order when possible."""
    cells = sorted((f for f in fragments if f[0] == column), key=lambda f: (f[1], f[2]))
    texts = [""] * len(rows)
    if len(cells) == len(rows):
        for index, cell in enumerate(cells):
            texts[index] = cell[3]
        return texts
    for cell in cells:
        centre = (cell[1] + cell[2]) / 2
        best, best_overlap = 0, -1.0
        for index, (start, end) in enumerate(rows):
            overlap = min(cell[2], end) - max(cell[1], start)
            if overlap > best_overlap:
                best, best_overlap = index, overlap
        texts[best] = clean((texts[best] + " " + cell[3]).strip())
    return texts


def text_blocks(page, top: float, bottom: float):
    blocks = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[:5]
        if y1 <= top or y0 >= bottom:
            continue
        text = clean(text)
        if text:
            blocks.append((x0, y0, x1, y1, text))
    return blocks


def table_area(page) -> tuple[float, float]:
    rules = page_rules(page)
    if rules:
        return rules[0] + 1, rules[1]
    # No ruling lines (e.g. scan page 31): cut at the footnote below the table.
    bottom = page.rect.height - 60
    for line in text_lines(page, 0, page.rect.height):
        if line[0] < 130 and line[1] > 500:
            bottom = line[1]
            break
    return 150.0, bottom


def column_boundaries(page, top: float, bottom: float) -> list[float]:
    """Seven internal boundaries, snapped from rules to text-column starts."""
    expected = rule_columns(page)
    starts = []
    for line in text_lines(page, top, bottom):
        if PAGE_LEFT < line[0] < 480:
            starts.append(round(line[0], 1))
    starts.sort()
    clusters: list[list[float]] = []
    for x in starts:
        if clusters and x - clusters[-1][-1] <= 8:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    centres = [min(cluster) for cluster in clusters if len(cluster) >= 3]
    snapped = []
    for boundary in expected:
        candidates = [centre - 2.0 for centre in centres if abs(centre - 2.0 - boundary) <= 25]
        snapped.append(min(candidates, key=lambda c: abs(c - boundary)) if candidates else boundary)
    return snapped


def column_of(x0: float, x1: float, boundaries: list[float]) -> int:
    centre = (x0 + x1) / 2
    index = 0
    for boundary in boundaries:
        if centre > boundary:
            index += 1
    return index


def cluster_rows(blocks, tolerance: float = 8.0) -> list[float]:
    tops = sorted(block[1] for block in blocks)
    starts: list[float] = []
    for top in tops:
        if not starts or top - starts[-1] > tolerance:
            starts.append(top)
    return starts


def merge_close(cells: list[tuple], gap: float = 5.0) -> list[tuple]:
    """Merge block fragments that are pieces of the same cell."""
    merged: list[list] = []
    for cell in sorted(cells, key=lambda c: c[1]):
        if merged and cell[1] - merged[-1][2] <= gap:
            merged[-1][2] = max(merged[-1][2], cell[2])
            merged[-1][3] = clean(merged[-1][3] + " " + cell[3])
        else:
            merged.append(list(cell))
    return [tuple(cell) for cell in merged]


def assemble_page(page, page_no: int, valid_sections: set[str]) -> list[dict]:
    if not schedule_page(page):
        return []
    top, bottom = table_area(page)
    boundaries = column_boundaries(page, top, bottom)
    fragments = cell_fragments(page, top, bottom, boundaries)
    if not fragments:
        return []

    # One cell per table row in the offence column.  Adjacent fragments less
    # than a line apart are the same cell split by the text layer.
    offence = merge_close([f for f in fragments if f[0] == 1])
    if not offence:
        return []
    # The repeated header's "Offence." cell is the first row on the page.
    if "offence" in offence[0][3].casefold():
        offence = offence[1:]
    if not offence:
        return []
    header_cut = offence[0][1] - 2
    fragments = [f for f in fragments if f[1] >= header_cut]
    row_starts = [f[1] for f in offence]
    row_ends = row_starts[1:] + [bottom]
    rows = list(zip(row_starts, row_ends))

    # Section numbers are printed in reading order and may share one line
    # ("406. 407. 408."); consume one number per offence cell, carrying the
    # last number into following variant rows.
    number_tokens: list[tuple[float, str]] = []
    label_y = None
    for line in text_lines(page, top, bottom):
        if column_of(line[0], line[2], boundaries) != 0:
            continue
        if re.fullmatch(r"Section\.?", line[4], re.I):
            label_y = line[3]
            continue
        tokens = re.findall(r"\d{1,3}[A-Z]?", line[4].upper())
        if not tokens:
            continue
        if re.fullmatch(r"[1-8]\.?", line[4]) and label_y and line[3] <= label_y + 10:
            continue
        for token in tokens:
            number_tokens.append((line[1], token if token in valid_sections else ""))

    sections: list[str] = []
    pointer = 0
    current = None
    for start in row_starts:
        if pointer < len(number_tokens) and number_tokens[pointer][0] <= start + 2:
            current = number_tokens[pointer][1] or current
            pointer += 1
        sections.append(current or "")

    columns = {column: assign_cells(fragments, column, rows) for column in range(2, 8)}
    output = []
    for index, start in enumerate(row_starts):
        if not sections[index]:
            continue
        cells = [""] * 8
        cells[0] = sections[index]
        cells[1] = offence[index][3]
        for column in range(2, 8):
            cells[column] = columns[column][index]
        output.append({
            "section_no": sections[index],
            "section_cell": cells[0],
            "cells": cells,
            "page": page_no,
            "row_index": index + 1,
        })
    return output


def resolve_ditto(raw_cells: list[str], previous: dict[int, str]) -> list[str]:
    """Resolve "Ditto" against the cell above in the same column."""
    resolved = list(raw_cells)
    for index in range(1, 8):
        value = resolved[index].strip()
        if value.casefold().rstrip(".") in DITTO:
            resolved[index] = previous.get(index, "")
        elif value:
            previous[index] = value
    return resolved


def normalise_bailable(text: str):
    low = clean(text).casefold()
    if "not bailable" in low or "non-bailable" in low or "non bailable" in low:
        return False
    if "bailable" in low:
        return True
    return None


def normalise_cognizable(text: str):
    low = clean(text).casefold()
    if "may arrest without warrant" in low:
        return True
    if "shall not arrest without warrant" in low or "not arrest without warrant" in low:
        return False
    return None


def build_rows(pages_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    previous: dict[int, str] = {}
    for record in pages_rows:
        raw_cells = record["cells"]
        cells = resolve_ditto(raw_cells, previous)
        section = record["section_no"]
        rows.append({
            "procedure_id": f"{section}::p{record['page']}r{record['row_index']}",
            "penal_section": section,
            "offence_or_variant": cells[1] or None,
            "cognizable": normalise_cognizable(cells[2]),
            "arrest_rule_text": cells[2] or None,
            "process": cells[3] or None,
            "bailable": normalise_bailable(cells[4]),
            "compoundable": cells[5] or None,
            "punishment": cells[6] or None,
            "trial_court": cells[7] or None,
            "source_url": PDF_URL,
            "source_page": record["page"],
            "source_type": "official_schedule_pdf_geometry_extraction",
            "amendment_source_url": AMENDMENT_URL,
            "verified": False,
            "parser_status": "auto_extracted",
            "raw_schedule_cells": {str(i + 1): raw_cells[i] for i in range(8)},
        })
    return apply_2026_schedule_amendments(rows)


def apply_2026_schedule_amendments(rows: list[dict]) -> list[dict]:
    """Overlay Schedule II changes in CrPC (Amendment) Act, 2026, section 28."""
    patches = {
        "143": {"compoundable": "Compoundable"},
        "144": {"compoundable": "Not compoundable"},
        "325": {"process": "Warrant", "bailable": False},
    }
    seen = set()
    for row in rows:
        patch = patches.get(row["penal_section"])
        if patch:
            row.update(patch)
            row["amendment_applied"] = "Code of Criminal Procedure (Amendment) Act, 2026, section 28"
            seen.add(row["penal_section"])
    for section, patch in patches.items():
        if section not in seen:
            rows.append({
                "procedure_id": f"{section}::amendment_overlay_only",
                "penal_section": section,
                "offence_or_variant": None,
                "cognizable": None,
                "arrest_rule_text": None,
                "process": patch.get("process"),
                "bailable": patch.get("bailable"),
                "compoundable": patch.get("compoundable"),
                "punishment": None,
                "trial_court": None,
                "source_url": PDF_URL,
                "source_page": None,
                "source_type": "official_amendment_overlay_only",
                "amendment_source_url": AMENDMENT_URL,
                "amendment_applied": "Code of Criminal Procedure (Amendment) Act, 2026, section 28",
                "verified": False,
                "parser_status": "base_schedule_row_not_extracted",
                "raw_schedule_cells": None,
            })
    return rows


def write_outputs(rows: list[dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Bangladesh Laws CrPC Schedule II PDF with 2026 Act section 28 overlay",
        "source_url": PDF_URL,
        "amendment_source_url": AMENDMENT_URL,
        "generated_automatically": True,
        "all_rows_unverified": True,
        "extraction_method": "block-top rows, line-level column bucketing, rules snapped to text",
        "rows": {row["procedure_id"]: row for row in rows},
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["procedure_id", "penal_section", "offence_or_variant", "cognizable",
              "arrest_rule_text", "process", "bailable", "compoundable", "punishment",
              "trial_court", "source_page", "amendment_applied", "verified",
              "parser_status", "raw_schedule_cells"]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            out["raw_schedule_cells"] = json.dumps(row.get("raw_schedule_cells"), ensure_ascii=False)
            writer.writerow(out)


def main() -> int:
    sections_path = ROOT / "data" / "clean" / "sections.jsonl"
    if not sections_path.exists():
        raise SystemExit("Run Stage 1 first; data/clean/sections.jsonl is missing")
    valid_sections = set()
    for line in sections_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("act_id") == 11:
            number = str(record.get("section_no", "")).upper()
            if re.fullmatch(r"\d{1,3}[A-Z]?", number):
                valid_sections.add(number)

    download_pdf()
    document = pymupdf.open(PDF_PATH)
    page_rows = []
    for page_index, page in enumerate(document):
        page_rows.extend(assemble_page(page, page_index + 1, valid_sections))
    rows = build_rows(page_rows)
    write_outputs(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["parser_status"]] = counts.get(row["parser_status"], 0) + 1
    print(json.dumps({
        "pages": document.page_count,
        "rows": len(rows),
        "distinct_penal_sections": len({row["penal_section"] for row in rows}),
        "verified": 0,
        "parser_status": counts,
        "procedure_json": str(OUT_JSON),
        "review_csv": str(OUT_CSV),
    }, indent=2))
    print("All extracted facts are candidates, not legally verified findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
