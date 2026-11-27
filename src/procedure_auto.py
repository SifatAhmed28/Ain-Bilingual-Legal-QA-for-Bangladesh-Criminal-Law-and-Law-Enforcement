"""Automatically extract CrPC Schedule II candidates with page provenance.

This is a best-effort parser. Every extracted row remains unverified; the
source page and raw cells are retained so the output is auditable.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import requests

try:
    import pymupdf
except ImportError:  # compatibility with older PyMuPDF installs
    import fitz as pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_URL = "https://bdlaws.minlaw.gov.bd/upload/act/2025-08-17-15-16-07-75___170825-Schedule.pdf"
AMENDMENT_URL = "https://bdlaws.minlaw.gov.bd/act-print-1640/section-print-58332.html"
PDF_PATH = ROOT / "data" / "raw" / "bdlaws" / "crpc_schedule_ii_2025.pdf"
OUT_JSON = ROOT / "data" / "clean" / "procedure.json"
OUT_CSV = ROOT / "data" / "clean" / "procedure_auto_candidates.csv"


def download_pdf() -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 100_000:
        return
    response = requests.get(PDF_URL, timeout=180)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError("Official schedule URL did not return a PDF")
    PDF_PATH.write_bytes(response.content)


def clean(value) -> str:
    text = str(value or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def section_numbers(cell: str, valid_sections: set[str]) -> list[str]:
    # Strip common footnote notation before identifying section numbers.
    text = re.sub(r"\d+\s*\[\s*(?:\*+|\d+)\s*\]", " ", cell)
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", " ", text)
    candidates = re.findall(r"(?<!\w)(\d{1,3}[A-Z]?)(?!\w)", text.upper())
    found=[]
    for value in candidates:
        if value in valid_sections and value not in found:
            found.append(value)
    return found


def tri_state(text: str, positive: tuple[str, ...], negative: tuple[str, ...]):
    low=clean(text).casefold()
    no=any(x in low for x in negative)
    remainder=low
    for phrase in negative:
        remainder=remainder.replace(phrase, " ")
    yes=any(x in remainder for x in positive)
    if yes and not no: return True
    if no and not yes: return False
    return None


def resolve_ditto(value: str, previous: str | None) -> str:
    return previous if clean(value).casefold() in {"ditto", "do.", "do"} and previous else clean(value)


def extract_rows(valid_sections: set[str]) -> list[dict]:
    download_pdf()
    document=pymupdf.open(PDF_PATH)
    rows=[]
    previous_by_column={}
    for page_index,page in enumerate(document):
        page_text=page.get_text("text")
        normalized_page_text=re.sub(r"\s+"," ",page_text).casefold()
        if "whether the police may arrest without warrant" not in normalized_page_text:
            continue
        try:
            tables=page.find_tables().tables
            if not tables:
                tables=page.find_tables(strategy="text").tables
        except Exception:
            try: tables=page.find_tables(strategy="text").tables
            except Exception: tables=[]
        for table in tables:
            try: matrix=table.extract()
            except Exception: continue
            for row_index,cells in enumerate(matrix):
                cells=[clean(x) for x in cells]
                if len(cells)<8: continue
                nums=section_numbers(cells[0],valid_sections)
                if not nums: continue
                raw={str(i+1):cells[i] for i in range(8)}
                resolved=[cells[0]]
                for col in range(1,8):
                    value=cells[col]
                    if clean(value).casefold() in {"ditto","do.","do"}:
                        value=previous_by_column.get(col,value)
                    elif value:
                        previous_by_column[col]=value
                    resolved.append(value)
                # Most rows are one offence per section. Grouped rows are
                # retained, but their parse status flags any uncertain split.
                status="auto_extracted" if len(nums)==1 else "grouped_sections_needs_alignment"
                for n,section in enumerate(nums):
                    variant_id=f"{section}::pdfp{page_index+1}r{row_index+1}"
                    offense=resolved[1]
                    if len(nums)>1: offense=f"{offense} [grouped schedule row; section {section}]"
                    arrest=resolved[2]
                    process=resolved[3]
                    bail=resolved[4]
                    compound=resolved[5]
                    punishment=resolved[6]
                    court=resolved[7]
                    record={
                        "procedure_id":variant_id,
                        "penal_section":section,
                        "offence_or_variant":offense or None,
                        "cognizable":tri_state(arrest,("may arrest without warrant",),("shall not arrest without warrant",)),
                        "process":process or None,
                        "bailable":tri_state(bail,("bailable",),("not bailable","non-bailable")),
                        "compoundable":compound or None,
                        "punishment":punishment or None,
                        "trial_court":court or None,
                        "source_url":PDF_URL,
                        "source_page":page_index+1,
                        "source_type":"official_schedule_pdf_auto_extraction",
                        "amendment_source_url":AMENDMENT_URL,
                        "verified":False,
                        "parser_status":status,
                        "raw_schedule_cells":raw,
                    }
                    rows.append(record)
    return apply_2026_schedule_amendments(rows)


def apply_2026_schedule_amendments(rows: list[dict]) -> list[dict]:
    """Overlay Schedule II changes in CrPC Amendment Act 2026, section 28."""
    patches={
        "143":{"compoundable":"Compoundable"},
        "144":{"compoundable":"Not compoundable"},
        "325":{"process":"Warrant","bailable":False},
    }
    seen=set()
    for row in rows:
        patch=patches.get(row["penal_section"])
        if patch:
            row.update(patch)
            row["amendment_applied"]="Code of Criminal Procedure (Amendment) Act, 2026, section 28"
            seen.add(row["penal_section"])
    for section,patch in patches.items():
        if section not in seen:
            rows.append({
                "procedure_id":f"{section}::amendment_overlay_only",
                "penal_section":section,
                "offence_or_variant":None,
                "cognizable":None,
                "process":patch.get("process"),
                "bailable":patch.get("bailable"),
                "compoundable":patch.get("compoundable"),
                "punishment":None,
                "trial_court":None,
                "source_url":PDF_URL,
                "source_page":None,
                "source_type":"official_amendment_overlay_only",
                "amendment_source_url":AMENDMENT_URL,
                "amendment_applied":"Code of Criminal Procedure (Amendment) Act, 2026, section 28",
                "verified":False,
                "parser_status":"base_schedule_row_not_extracted",
                "raw_schedule_cells":None,
            })
    return rows


def write_outputs(rows: list[dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True)
    payload={"source":"Bangladesh Laws CrPC Schedule II PDF with 2026 Act section 28 overlay",
             "source_url":PDF_URL,"amendment_source_url":AMENDMENT_URL,
             "generated_automatically":True,"all_rows_unverified":True,
             "rows":{r["procedure_id"]:r for r in rows}}
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    fields=["procedure_id","penal_section","offence_or_variant","cognizable","process","bailable","compoundable","punishment","trial_court","source_url","source_page","amendment_source_url","amendment_applied","verified","parser_status","raw_schedule_cells"]
    with OUT_CSV.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows:
            out={k:row.get(k) for k in fields}
            out["raw_schedule_cells"]=json.dumps(row.get("raw_schedule_cells"),ensure_ascii=False)
            writer.writerow(out)


def main() -> int:
    sections_path=ROOT/"data"/"clean"/"sections.jsonl"
    if not sections_path.exists():
        raise SystemExit("Run Stage 1 first; data/clean/sections.jsonl is missing")
    valid_sections=set()
    for line in sections_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        record=json.loads(line)
        if record.get("act_id")==11:
            no=str(record.get("section_no","")).upper()
            if re.fullmatch(r"\d{1,3}[A-Z]?",no): valid_sections.add(no)
    rows=extract_rows(valid_sections)
    write_outputs(rows)
    counts={}
    for row in rows: counts[row["parser_status"]]=counts.get(row["parser_status"],0)+1
    print(json.dumps({"rows":len(rows),"verified":0,"parser_status":counts,
                      "procedure_json":str(OUT_JSON),"review_csv":str(OUT_CSV)},indent=2))
    print("All extracted facts are candidates, not legally verified findings.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
