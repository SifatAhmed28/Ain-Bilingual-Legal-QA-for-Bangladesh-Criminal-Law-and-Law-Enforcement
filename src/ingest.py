"""Stage 1 ingestion.

Sources
-------
* bdlaws (canonical, current to 2026): per-section xhtml with titles, chapters
  and footnotes.  Cached under ``data/raw/bdlaws/html`` so later runs are
  offline.
* BLAD (Kaggle): per-act JSON with section content, used as offline fallback.
* UKIL-DB-EN: English section names + footnotes, used to enrich/verify.
* bdlaws chronological index: authoritative repealed flag for every act.

Output
------
``data/raw/acts/act-<id>.json`` for every whitelisted act.  Parsing into
final schema happens in :mod:`src.preprocess`.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from lxml import html as LH

from . import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def normalize_digits(text: str) -> str:
    return text.translate(C.BN_DIGITS)


def norm_space(text: str) -> str:
    if not text:
        return ""
    text = (
        text.replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\uf02d", "-")
        .replace("\uf0b7", "-")
    )
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def decode_html(raw: bytes) -> str:
    """bdlaws serves a mix of UTF-16 (with BOM) and UTF-8 pages."""
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16")
    if raw[:400].count(b"\x00") > 40:
        return raw.decode("utf-16-le", errors="ignore")
    return raw.decode("utf-8-sig", errors="ignore")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def fetch_cached(
    url: str,
    cache_path: Path,
    *,
    force: bool = False,
    offline: bool = False,
    session: requests.Session | None = None,
) -> tuple[str | None, str]:
    """Return (html, state) where state is cache|fetched|offline-miss|error."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        return decode_html(cache_path.read_bytes()), "cache"
    if offline:
        return None, "offline-miss"

    session = session or requests.Session()
    last_error = ""
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=75, headers={"User-Agent": C.USER_AGENT})
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
            time.sleep(C.FETCH_DELAY_SECONDS)
            return decode_html(resp.content), "fetched"
        except Exception as exc:  # noqa: BLE001 - network boundary
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    print(f"  ! fetch failed {url} ({last_error})")
    return None, "error"


# ---------------------------------------------------------------------------
# bdlaws chronological index
# ---------------------------------------------------------------------------

ACT_LINK_RE = re.compile(
    r"<a[^>]+href=['\"]/act-(\d+)\.html['\"][^>]*>(.*?)</a>", re.S | re.I
)


def parse_chronological_index(html_text: str) -> dict[int, dict[str, Any]]:
    entries: dict[int, dict[str, Any]] = {}
    for act_id_s, inner in ACT_LINK_RE.findall(html_text):
        title = norm_space(re.sub(r"<[^>]+>", " ", inner))
        if not title:
            continue
        act_id = int(act_id_s)
        if act_id in entries:
            continue  # first link is the act name; later links may be years
        repealed = bool(
            re.search(r"\[\s*repealed\s*\]|\[\s*রহিত\s*\]", title, re.I)
        )
        entries[act_id] = {
            "act_id": act_id,
            "title": title,
            "repealed": repealed,
        }
    return entries


def load_or_fetch_index(
    *, force: bool = False, offline: bool = False
) -> dict[int, dict[str, Any]]:
    path = C.RAW_BDLAWS / "index.json"
    if path.exists() and not force:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}

    html_text, state = fetch_cached(
        C.BDLAWS_CHRONO,
        C.RAW_BDLAWS_HTML / "_chronological_index.html",
        force=force,
        offline=offline,
    )
    if html_text is None:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {int(k): v for k, v in raw.items()}
        raise RuntimeError("bdlaws index unavailable and no cache present")

    entries = parse_chronological_index(html_text)
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"  index: {len(entries)} acts ({state})")
    return entries


# ---------------------------------------------------------------------------
# bdlaws act page
# ---------------------------------------------------------------------------

def _clean_node_text(node) -> str:
    """text_content() without footnote superscripts.

    Footnote spans wrap only the superscript marker; the bracketed amended
    wording (e.g. ``[Penal Code]``) is a *tail* of that span.  Deleting the
    span would delete the legal text with it, so only the superscript text is
    cleared.
    """
    import copy

    clone = copy.deepcopy(node)
    for bad in clone.xpath(".//script | .//style"):
        bad.getparent().remove(bad)
    for sup in clone.xpath(".//sup"):
        for sub in sup.iter():
            sub.text = ""
    return norm_space(clone.text_content())


def _footnote_number(text: str) -> int | None:
    m = re.match(r"\s*([0-9০-৯]{1,4})", normalize_digits(text))
    return int(m.group(1)) if m else None


def parse_bdlaws_page(html_text: str, act_id: int) -> dict[str, Any]:
    doc = LH.fromstring(html_text)

    title = ""
    subtitle = ""
    h3 = doc.xpath("//div[contains(@class,'text-center')]//h3")
    if h3:
        title = _clean_node_text(h3[0])
        parent = h3[0].getparent()
        if parent is not None:
            subtitle = _clean_node_text(parent)
    if not title:
        title = norm_space(" ".join(doc.xpath("//title/text()")))

    # "( ২০২৬ সনের ৮১ নং আইন ) [ ১০ এপ্রিল, ২০২৬ ]" / "(Act No. XLV of 1860)"
    act_no = None
    m = re.search(r"\(\s*([^()]{1,40}?)\s+নং\s+(আইন|অধ্যাদেশ)\s*\)", subtitle)
    if m:
        act_no = normalize_digits(m.group(1)).strip()
    else:
        m = re.search(r"Act No\.?\s*([^()]+?)\s+of\s+(\d{4})", subtitle, re.I)
        if m:
            act_no = norm_space(m.group(1))

    act_year = None
    m = re.search(r"(\d{4})\s*\)?\s*$", subtitle)
    if m:
        act_year = int(m.group(1))
    if act_year is None:
        m = re.search(r"(1[6-9]\d{2}|20\d{2})\s*$", title)
        if m:
            act_year = int(m.group(1))

    enactment_date = None
    m = re.search(r"\[\s*([^\]]{4,60})\s*\]", subtitle)
    if m:
        enactment_date = norm_space(m.group(1))

    footnotes: list[dict[str, Any]] = []
    for li in doc.xpath('//li[contains(@class,"footnoteList")]'):
        txt = norm_space(li.text_content())
        if not txt:
            continue
        footnotes.append({"n": _footnote_number(txt), "text": txt})

    sections: list[dict[str, Any]] = []
    chapters: list[dict[str, str]] = []
    chapter_no = ""
    chapter_name = ""
    order = 0

    # document order: chapter groups and section rows
    nodes = doc.xpath(
        '//div[contains(@class,"act-chapter-group")]'
        ' | //div[contains(@class,"row") and contains(@class,"lineremove")]'
    )
    for node in nodes:
        if "act-chapter-group" in (node.get("class") or ""):
            nos = node.xpath('.//p[contains(@class,"act-chapter-no")]')
            names = node.xpath('.//p[contains(@class,"act-chapter-name")]')
            chapter_no = norm_space(nos[0].text_content()) if nos else ""
            chapter_name = norm_space(names[0].text_content()) if names else ""
            if chapter_no or chapter_name:
                chapters.append(
                    {"no": chapter_no, "name": chapter_name}
                )
            continue

        details = node.xpath('.//div[contains(@class,"txt-details")]')
        if not details:
            continue
        head_nodes = node.xpath('.//div[contains(@class,"txt-head")]')
        head = _clean_node_text(head_nodes[0]) if head_nodes else ""
        body_node = details[0]

        markers: list[int] = []
        for sup in body_node.xpath(".//sup"):
            n = _footnote_number(sup.text_content())
            if n is not None:
                markers.append(n)

        text = _clean_node_text(body_node)
        if not text and not head:
            continue
        sections.append(
            {
                "order": order,
                "chapter_no": chapter_no,
                "chapter_name": chapter_name,
                "title": head,
                "text": text,
                "footnote_markers": sorted(set(markers)),
            }
        )
        order += 1

    return {
        "act_id": act_id,
        "url": C.BDLAWS_PRINT.format(act_id=act_id),
        "fetched_at": now_iso(),
        "title": title,
        "subtitle": subtitle,
        "act_no": act_no,
        "year": act_year,
        "enactment_date": enactment_date,
        "chapters": chapters,
        "sections": sections,
        "footnotes": footnotes,
    }


# ---------------------------------------------------------------------------
# BLAD / UKIL / CSV loaders
# ---------------------------------------------------------------------------

def load_blad_act(act_id: int) -> dict[str, Any] | None:
    path = C.BLAD_ACTS_DIR / f"act-print-{act_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "title": data.get("act_title"),
        "act_no": data.get("act_no"),
        "year": data.get("act_year"),
        "publication_date": data.get("publication_date"),
        "sections": data.get("sections", []),
    }


def load_blad_flags() -> dict[int, dict[str, Any]]:
    flags: dict[int, dict[str, Any]] = {}
    if not C.BLAD_ACT_LIST.exists():
        return flags
    with C.BLAD_ACT_LIST.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            m = re.search(r"act-print-(\d+)", row.get("full_text") or "")
            if not m:
                continue
            flags[int(m.group(1))] = {
                "title": row.get("Act Title"),
                "act_no": row.get("Act No"),
                "year": row.get("Act Year"),
                "repealed": (row.get("is_repealed") or "").strip().lower()
                in {"true", "1", "yes"},
            }
    return flags


def load_ukil_acts(wanted: set[int]) -> dict[int, dict[str, Any]]:
    """Stream-read the 110 MB UKIL file once and keep only wanted acts."""
    found: dict[int, dict[str, Any]] = {}
    if not C.UKIL_FILE.exists():
        return found
    data = json.loads(C.UKIL_FILE.read_text(encoding="utf-8"))
    for entry in data:
        act = entry.get("act") or {}
        act_id = act.get("id")
        if act_id in wanted:
            found[int(act_id)] = {
                "name": act.get("name"),
                "published_date": act.get("published_date"),
                "repealed": bool(act.get("repelled")),
                "sections": act.get("sections", []),
                "footnotes": act.get("lower_text", []),
            }
    return found


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(x) for x in re.findall(r"\d+", value)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="never hit the network")
    parser.add_argument("--force", action="store_true", help="ignore HTML cache")
    parser.add_argument("--only", help="comma-separated act ids")
    parser.add_argument("--no-bdlaws", action="store_true", help="skip bdlaws pages")
    args = parser.parse_args(argv)

    C.RAW_ACTS.mkdir(parents=True, exist_ok=True)
    C.RAW_BDLAWS_HTML.mkdir(parents=True, exist_ok=True)

    only = parse_ids(args.only)
    specs = [a for a in C.ACTS if only is None or a.act_id in only]
    if not specs:
        print("no acts selected")
        return 1

    print(f"[ingest] {len(specs)} acts selected")
    index = load_or_fetch_index(force=args.force, offline=args.offline)
    blad_flags = load_blad_flags()
    ukil = load_ukil_acts({a.act_id for a in specs})

    session = requests.Session()
    manifest: list[dict[str, Any]] = []
    missing: list[int] = []

    for spec in specs:
        print(f"  - {spec.act_id:4d} {spec.code:22s} {spec.name_en[:58]}")
        bdlaws = None
        state = "skipped"
        if not args.no_bdlaws:
            html_text, state = fetch_cached(
                C.BDLAWS_PRINT.format(act_id=spec.act_id),
                C.RAW_BDLAWS_HTML / f"act-{spec.act_id}.html",
                force=args.force,
                offline=args.offline,
                session=session,
            )
            if html_text:
                try:
                    bdlaws = parse_bdlaws_page(html_text, spec.act_id)
                except Exception as exc:  # noqa: BLE001 - malformed page
                    print(f"    ! parse failed: {type(exc).__name__}: {exc}")
                    bdlaws = None

        blad = load_blad_act(spec.act_id)
        if bdlaws is None and blad is None:
            missing.append(spec.act_id)
            print(f"    ! no bdlaws page and no BLAD record (state={state})")

        index_entry = index.get(spec.act_id)
        if index_entry is None:
            index_entry = next(
                (
                    v
                    for v in index.values()
                    if norm_space(v.get("title", ""))
                    .lower()
                    .startswith(spec.name_en.lower()[:24])
                ),
                None,
            )

        record = {
            "act_id": spec.act_id,
            "code": spec.code,
            "name_en": spec.name_en,
            "name_bn": spec.name_bn,
            "category": spec.category,
            "audit_only": spec.audit_only,
            "generated_at": now_iso(),
            "sources": {
                "bdlaws": bdlaws,
                "bdlaws_state": state,
                "blad": blad,
                "blad_list": blad_flags.get(spec.act_id),
                "ukil": ukil.get(spec.act_id),
                "index": index_entry,
            },
        }
        out = C.RAW_ACTS / f"act-{spec.act_id}.json"
        out.write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        html_path = C.RAW_BDLAWS_HTML / f"act-{spec.act_id}.html"
        manifest.append(
            {
                "act_id": spec.act_id,
                "code": spec.code,
                "bdlaws_state": state,
                "bdlaws_sections": len(bdlaws["sections"]) if bdlaws else 0,
                "blad_sections": len(blad["sections"]) if blad else 0,
                "ukil_sections": len((ukil.get(spec.act_id) or {}).get("sections", [])),
                "html_sha256": hashlib.sha256(html_path.read_bytes()).hexdigest()
                if html_path.exists()
                else None,
            }
        )

    (C.RAW_ACTS / "_manifest.json").write_text(
        json.dumps(
            {"generated_at": now_iso(), "acts": manifest, "missing": missing},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"[ingest] wrote {len(manifest)} raw act records to {C.RAW_ACTS}")
    if missing:
        print(f"[ingest] missing sources for act ids: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
