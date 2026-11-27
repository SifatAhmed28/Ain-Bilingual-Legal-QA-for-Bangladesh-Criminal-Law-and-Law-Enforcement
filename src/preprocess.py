"""Stage 1 preprocessing: raw sources -> ``data/clean/sections.jsonl``.

What happens here
-----------------
1. Pick the best section structure per act (bdlaws > BLAD > UKIL).
2. Recover section numbers (English and Bengali), merge continuation rows and
   attach chapter headings.
3. Join languages by ``sec_uid``: any English text and any Bangla text for the
   same act + section number end up in one record.
4. Drop repealed acts/sections, detect amendment dates, attach footnotes.
5. Emit sections.jsonl, acts.jsonl, dropped.jsonl, coverage.csv and a report.

The rule for missing translations is deliberate: no machine translation is
invented.  ``text_bn`` is null when no official Bangla text exists, and the
gap is reported so the retrieval stage can use a multilingual model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from . import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BN_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
BN_SUFFIX_RE = re.compile(r"([০-৯]{1,3})([ক-হ])")

MAX_YEAR = 2026

# ---------------------------------------------------------------------------
# basic normalisation
# ---------------------------------------------------------------------------

def norm(text: str | None) -> str:
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
    # Remaining private-use glyphs are Word bullet artefacts.
    text = re.sub(r"[\uf000-\uf0ff]", " ", text)
    # One bdlaws act stores cp1252-misdecoded UTF-8 (e.g. "ÔÇö" for an em dash).
    for bad, good in (
        ("ÔÇö", "—"),
        ("ÔÇô", "–"),
        ("ÔÇ£", '"'),
        ("ÔÇ¥", '"'),
        ("ÔÇÿ", "'"),
        ("ÔÇÖ", "'"),
        ("ÔÇª", "…"),
        ("ÔÇ¿", "?"),
    ):
        if bad in text:
            text = text.replace(bad, good)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def bengali_to_ascii(text: str) -> str:
    """Convert Bengali digits and common section suffixes (৯ক -> 9A)."""
    # Suffix mapping must run while the digits are still Bengali, otherwise
    # BN_SUFFIX_RE cannot anchor on them.
    text = BN_SUFFIX_RE.sub(
        lambda m: m.group(1) + C.BN_SECTION_SUFFIX.get(m.group(2), ""), text
    )
    return text.translate(C.BN_DIGITS)


def detect_lang(text: str) -> str:
    if not text:
        return "en"
    bn = len(BN_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = bn + latin
    if total == 0:
        return "en"
    if bn >= 8 and bn / total >= 0.20:
        return "bn"
    if latin >= 8 and bn / total <= 0.05:
        return "en"
    return "mixed"


# ---------------------------------------------------------------------------
# section numbers
# ---------------------------------------------------------------------------

EN_NUM_RE = re.compile(r"^(\d{1,3}[A-Z]{0,2})\s*(?:[.\-–—।৷]|\s)\s*")
BN_NUM_RE = re.compile(r"^([০-৯]{1,3}[ক-হA-Z]?)\s*(?:[।৷.\-–—]|\s)\s*")
LEADING_FN_RE = re.compile(r"^\s*(?:\d{1,3}|[০-৯]{1,3})\s*\[")
PREAMBLE_RE = re.compile(r"^(whereas|যেহেতু|যেহেতু\b)", re.I)


def strip_leading_footnote_markers(text: str) -> str:
    text = text.lstrip()
    changed = True
    while changed:
        changed = False
        m = LEADING_FN_RE.match(text)
        if m:
            text = text[m.end():].lstrip()
            changed = True
    return text


def parse_section_number(text: str) -> str | None:
    """Return canonical section number, or None for preamble/continuation."""
    if not text:
        return None
    probe = strip_leading_footnote_markers(norm(text))
    probe = re.sub(r"^[.\[\]\s|]+", "", probe)
    probe = bengali_to_ascii(probe)

    m = EN_NUM_RE.match(probe)
    if m:
        num = m.group(1).upper()
        # "1 A" (space-separated) would be caught above only with a digit start
        return num

    m = BN_NUM_RE.match(probe)
    if m:
        return bengali_to_ascii(m.group(1)).upper()

    return None


def is_continuation(text: str) -> bool:
    probe = strip_leading_footnote_markers(norm(text))
    return probe.startswith("(") or probe.startswith("（")


def resolve_section_number(text: str, known: set[str] | None = None) -> str | None:
    """Parse a section number, repairing a footnote digit glued to a suffix.

    Some sources write inserted section 52A as ``1452A.`` (footnote 14 + 52A)
    or 9A as ``139ক।`` (footnote 13 + ৯ক).  When the naive parse is not a
    section we already know, retry without 1-2 leading digits, but only for
    letter-suffixed sections, where this corruption actually occurs.
    """
    num = parse_section_number(text)
    if num is None or not known or num in known:
        return num
    if not re.search(r"[A-Z]$", num):
        return num
    probe = norm(text)
    for cut in (2, 1):
        candidate = parse_section_number(probe[cut:])
        if candidate and candidate in known and re.search(r"[A-Z]$", candidate):
            return candidate
    return num


def section_sort_key(section_no: str) -> tuple:
    if section_no == "preamble":
        return (-1, 0, "")
    m = re.match(r"(\d+)([A-Z]*)$", section_no)
    if m:
        return (int(m.group(1)), 0, m.group(2))
    return (10**6, 0, section_no)


# ---------------------------------------------------------------------------
# per-source adapters
# ---------------------------------------------------------------------------

def clean_title(title: str | None) -> str:
    title = norm(title)
    title = strip_leading_footnote_markers(title)
    title = re.sub(r"^\s*\[\s*\d+\s*\]\s*", "", title)
    title = re.sub(r"\s*\[\s*\d+\s*\]\s*$", "", title)
    title = re.sub(r"^\d{1,3}\s*[.)]\s*", "", title)
    return title.strip(" .;:-")


def is_chapter_title(title: str) -> bool:
    if not title:
        return False
    if len(title) > 140:
        return False
    if title.isupper():
        return True
    return bool(
        re.match(r"^(part|chapter|of\s|প্রথম|দ্বিতীয়|তৃতীয়|চতুর্থ|পঞ্চম)", title, re.I)
    )


def _split_leading_prefix(text: str) -> tuple[str, str] | None:
    """Split "Some title 371. body" into ("Some title", "371. body").

    bdlaws occasionally leaks the previous section's title into the start of
    the next row's body (and puts the real title in ``txt-head``).
    """
    m = re.match(
        r"^(?P<prefix>[^\d।৷|]{3,150}?)\s+"
        r"(?P<num>\d{1,3}[A-Z]?|[০-৯]{1,3}[ক-হ]?)\s*[.\-–।৷]\s+"
        r"(?P<rest>.+)$",
        text,
        re.S,
    )
    if not m:
        return None
    return m.group("prefix").strip(" .;:-|"), norm(
        m.group("num") + ". " + m.group("rest")
    )


def sections_from_bdlaws(raw: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in raw.get("sections", []):
        text = norm(row.get("text"))
        text = re.sub(r"^[.\s|]+", "", text)
        # a row may start with the footnote bracket of an inserted section:
        # "[52A. ..." -> "52A. ..."
        text = re.sub(r"^\[(?=\s*[\d০-৯])", "", text)
        if not text:
            continue
        title = clean_title(row.get("title"))
        if parse_section_number(text) is None:
            split = _split_leading_prefix(text)
            if split:
                prefix, rebuilt = split
                if not title:
                    title = clean_title(prefix)
                text = rebuilt
        chapter = " ".join(
            x for x in [row.get("chapter_no"), row.get("chapter_name")] if x
        ).strip()
        out.append(
            {
                "section_no": parse_section_number(text),
                "title": title,
                "text": text,
                "chapter": chapter or None,
                "footnote_markers": row.get("footnote_markers") or [],
                "order": len(out),
            }
        )
    return out


def sections_from_plain(
    items: Iterable[dict[str, Any]],
    text_key: str,
    title_key: str,
    known: set[str] | None = None,
) -> list[dict[str, Any]]:
    """BLAD / UKIL adapter: merge continuation rows, recover chapters."""
    out: list[dict[str, Any]] = []
    current_chapter: str | None = None
    for item in items:
        text = strip_leading_footnote_markers(norm(item.get(text_key)))
        title = clean_title(item.get(title_key))
        if not text:
            continue
        num = resolve_section_number(text, known)
        if num:
            if is_chapter_title(title):
                current_chapter = title
            out.append(
                {
                    "section_no": num,
                    "title": "" if is_chapter_title(title) else title,
                    "text": text,
                    "chapter": current_chapter,
                    "footnote_markers": [
                        int(x) for x in re.findall(r"(?<![\dA-Za-z])(\d{1,3})\s*\[", text)
                    ],
                    "order": len(out),
                }
            )
            continue

        if PREAMBLE_RE.match(text) and not out:
            out.append(
                {
                    "section_no": "preamble",
                    "title": "Preamble",
                    "text": text,
                    "chapter": None,
                    "footnote_markers": [],
                    "order": 0,
                }
            )
            continue

        # continuation of the previous section
        if out:
            out[-1]["text"] = norm(out[-1]["text"] + " " + text)
            marks = {
                int(x) for x in re.findall(r"(?<![\dA-Za-z])(\d{1,3})\s*\[", text)
            }
            out[-1]["footnote_markers"] = sorted(
                set(out[-1]["footnote_markers"]) | marks
            )
        elif PREAMBLE_RE.match(text):
            out.append(
                {
                    "section_no": "preamble",
                    "title": "Preamble",
                    "text": text,
                    "chapter": None,
                    "footnote_markers": [],
                    "order": 0,
                }
            )
    for i, sec in enumerate(out):
        sec["order"] = i
    return out


def merge_number_rows(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge rows with no number (bdlaws sub-rows) into the previous section."""
    merged: list[dict[str, Any]] = []
    for sec in sections:
        if sec["section_no"] is None:
            if merged:
                merged[-1]["text"] = norm(merged[-1]["text"] + " " + sec["text"])
                merged[-1]["footnote_markers"] = sorted(
                    set(merged[-1]["footnote_markers"])
                    | set(sec["footnote_markers"])
                )
                if not merged[-1]["title"] and sec["title"]:
                    merged[-1]["title"] = sec["title"]
            elif PREAMBLE_RE.match(sec["text"]):
                sec["section_no"] = "preamble"
                sec["title"] = sec["title"] or "Preamble"
                merged.append(sec)
            continue
        if merged and merged[-1]["section_no"] == sec["section_no"]:
            merged[-1]["text"] = norm(merged[-1]["text"] + " " + sec["text"])
            merged[-1]["footnote_markers"] = sorted(
                set(merged[-1]["footnote_markers"]) | set(sec["footnote_markers"])
            )
            if not merged[-1]["title"] and sec["title"]:
                merged[-1]["title"] = sec["title"]
            continue
        merged.append(sec)
    for i, sec in enumerate(merged):
        sec["order"] = i
    return merged


def ukil_title_map(
    ukil: dict[str, Any] | None, known: set[str] | None = None
) -> dict[str, str]:
    """section number -> title (continuation titles appended).

    UKIL writes some inserted sections as ``1452A.`` — footnote 14 glued to
    section 52A.  When the plain parse fails we retry after dropping 1-2
    leading digits, but only accept the result if it is a section we know
    from the primary source.
    """
    mapping: dict[str, str] = {}
    if not ukil:
        return mapping

    last_num: str | None = None
    for item in ukil.get("sections", []):
        details = norm(item.get("details"))
        name = clean_title(item.get("name"))
        if not details:
            continue
        num = resolve_section_number(details, known)
        if num:
            last_num = num
            mapping[num] = name
        elif last_num and name and not is_chapter_title(name):
            mapping[last_num] = norm(mapping[last_num] + "; " + name)
    return mapping


def ukil_footnotes(ukil: dict[str, Any] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    if not ukil:
        return out
    for f in ukil.get("footnotes", []):
        f = norm(f)
        m = re.match(r"^\s*(\d{1,3})\s*(.*)$", f)
        if m:
            out[int(m.group(1))] = f
    return out


# ---------------------------------------------------------------------------
# amendment detection
# ---------------------------------------------------------------------------

EN_DATE_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"[,.\s]+(\d{4})\b",
    re.I,
)
BN_DATE_RE = re.compile(
    r"(\d{1,2})\s+"
    r"(জানুয়ারি|জানুয়ারি|ফেব্রুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)"
    r"[,.\s]+(\d{4})"
)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def extract_amendment(text: str, act_year: int) -> str | None:
    """Latest amendment date/year mentioned in the act's text or footnotes."""
    text = bengali_to_ascii(text)
    dates: list[date] = []
    for day, month_name, year in EN_DATE_RE.findall(text):
        month = C.EN_MONTHS.get(month_name.lower())
        if not month:
            continue
        try:
            d = date(int(year), month, int(day))
        except ValueError:
            continue
        if act_year <= d.year <= MAX_YEAR:
            dates.append(d)
    for day, month_name, year in BN_DATE_RE.findall(text):
        month = C.BN_MONTHS.get(month_name)
        if not month:
            continue
        try:
            d = date(int(year), month, int(day))
        except ValueError:
            continue
        if act_year <= d.year <= MAX_YEAR:
            dates.append(d)

    years = {
        int(y) for y in YEAR_RE.findall(text) if act_year <= int(y) <= MAX_YEAR
    }
    if dates:
        return max(dates).isoformat()
    if years:
        latest = max(years)
        if latest > act_year:
            return str(latest)
    return None


# ---------------------------------------------------------------------------
# assembling act records
# ---------------------------------------------------------------------------

def act_is_repealed(raw: dict[str, Any]) -> tuple[bool, str]:
    spec_audit = bool(raw.get("audit_only"))
    src = raw.get("sources", {})
    index = src.get("index") or {}
    blad = src.get("blad_list") or {}
    ukil = src.get("ukil") or {}
    name = raw.get("name_en", "")

    if index.get("repealed"):
        return True, "bdlaws chronological index marks [Repealed]"
    if blad.get("repealed"):
        return True, "BLAD filtered_act_list is_repealed=True"
    if ukil.get("repealed"):
        return True, "UKIL act.repelled=True"
    if src.get("bdlaws") and re.search(
        r"\[\s*(repealed|রহিত)\s*\]", src["bdlaws"].get("title") or "", re.I
    ):
        return True, "bdlaws page title marks [Repealed]"
    if spec_audit:
        return True, "whitelist marks act as repealed (audit_only)"
    if re.search(r"repeal", name, re.I):
        return True, "act name contains Repeal"
    return False, ""


def section_is_repealed(sec: dict[str, Any]) -> bool:
    text = norm(sec.get("text"))
    body = text.strip(" []")
    if re.fullmatch(
        r"(repealed|omitted)(\s+by\s+.*)?\.?", body, re.I | re.S
    ):
        return True
    if re.fullmatch(r"(রহিত|বিলুপ্ত)(\s+.*)?", body):
        return True
    # bdlaws prints a repealed section's body as "[***]" (sometimes with the
    # section number still in front, e.g. "5. [***]").
    if re.fullmatch(r"[\d০-৯]+[।.\s]*\[\s*\*{2,}\s*\]\s*", body):
        return True
    title = sec.get("title") or ""
    if re.search(r"\[\s*(repealed|রহিত)\s*\]", title, re.I) and len(text) < 160:
        return True
    return False


def choose_primary(raw: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    src = raw.get("sources", {})
    bdlaws = src.get("bdlaws")
    blad = src.get("blad")
    ukil = src.get("ukil")

    if bdlaws and len(bdlaws.get("sections", [])) >= 2:
        return "bdlaws", merge_number_rows(sections_from_bdlaws(bdlaws))
    if blad and blad.get("sections"):
        secs = sections_from_plain(
            blad["sections"], text_key="section_content", title_key="section_title"
        )
        return "blad", merge_number_rows(secs)
    if ukil and ukil.get("sections"):
        secs = sections_from_plain(
            ukil["sections"], text_key="details", title_key="name"
        )
        return "ukil", merge_number_rows(secs)
    return "none", []


def build_bilingual_buckets(
    raw: dict[str, Any],
    primary_name: str,
    primary_sections: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group section candidates by section_no and language, then pick best.

    Priority is bdlaws > BLAD > UKIL.  Today an act is monolingual, but this
    is the join point where a future Bangla/English parallel source lands.
    """
    src = raw.get("sources", {})
    priority = {"bdlaws": 1, "blad": 2, "ukil": 3}
    buckets: dict[str, dict[str, Any]] = {}

    def add(sec: dict[str, Any], source: str) -> None:
        sec_no = sec.get("section_no")
        if not sec_no:
            return
        lang = detect_lang(sec["text"])
        bucket = buckets.setdefault(
            sec_no,
            {
                "section_no": sec_no,
                "title": "",
                "text_en": None,
                "text_bn": None,
                "source_en": None,
                "source_bn": None,
                "chapter": None,
                "footnote_markers": set(),
                "order": sec.get("order", 10**6),
                "_title_priority": 99,
                "_en_priority": 99,
                "_bn_priority": 99,
            },
        )
        bucket["order"] = min(bucket["order"], sec.get("order", 10**6))
        bucket["chapter"] = bucket["chapter"] or sec.get("chapter")
        bucket["footnote_markers"] |= set(sec.get("footnote_markers") or [])

        title = sec.get("title") or ""
        if title and priority[source] < bucket["_title_priority"]:
            bucket["title"] = title
            bucket["_title_priority"] = priority[source]

        if lang == "bn":
            if priority[source] < bucket["_bn_priority"]:
                bucket["text_bn"] = sec["text"]
                bucket["source_bn"] = source
                bucket["_bn_priority"] = priority[source]
        elif lang == "en":
            if priority[source] < bucket["_en_priority"]:
                bucket["text_en"] = sec["text"]
                bucket["source_en"] = source
                bucket["_en_priority"] = priority[source]
        else:  # mixed script text: keep as Bangla if Bengali dominates
            if priority[source] < bucket["_bn_priority"]:
                bucket["text_bn"] = sec["text"]
                bucket["source_bn"] = source
                bucket["_bn_priority"] = priority[source]

    for sec in primary_sections:
        add(sec, primary_name)

    known = set(buckets)

    # Secondary monolingual sources only fill a language slot that is empty.
    if primary_name != "bdlaws" and src.get("bdlaws"):
        for sec in merge_number_rows(sections_from_bdlaws(src["bdlaws"])):
            add(sec, "bdlaws")
        known |= set(buckets)
    if primary_name != "blad" and src.get("blad"):
        for sec in merge_number_rows(
            sections_from_plain(
                src["blad"]["sections"],
                text_key="section_content",
                title_key="section_title",
                known=known,
            )
        ):
            add(sec, "blad")
    if primary_name != "ukil" and src.get("ukil"):
        for sec in merge_number_rows(
            sections_from_plain(
                src["ukil"]["sections"],
                text_key="details",
                title_key="name",
                known=known,
            )
        ):
            add(sec, "ukil")

    return buckets


def find_all_text(raw: dict[str, Any]) -> str:
    src = raw.get("sources", {})
    chunks: list[str] = []
    bdlaws = src.get("bdlaws") or {}
    chunks.extend(f.get("text", "") for f in bdlaws.get("footnotes", []))
    blad = src.get("blad") or {}
    chunks.extend(s.get("section_content", "") for s in blad.get("sections", []))
    ukil = src.get("ukil") or {}
    chunks.extend(ukil.get("footnotes", []))
    return "\n".join(chunks)


def build_act(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    act_id = raw["act_id"]
    name_en = raw["name_en"]
    name_bn = raw["name_bn"]
    act_year = int(
        (raw.get("sources", {}).get("bdlaws") or {}).get("year")
        or (raw.get("sources", {}).get("blad") or {}).get("year")
        or re.search(r"(1[6-9]\d{2}|20\d{2})", name_en).group(1)
    )
    amended = extract_amendment(find_all_text(raw), act_year)
    repealed, reason = act_is_repealed(raw)

    src = raw.get("sources", {})
    source_url = (src.get("bdlaws") or {}).get("url") or C.BDLAWS_PRINT.format(
        act_id=act_id
    )

    dropped: list[dict[str, Any]] = []
    if repealed:
        dropped.append(
            {
                "level": "act",
                "act_id": act_id,
                "act_name": name_en,
                "reason": reason,
                "sections": len((src.get("bdlaws") or {}).get("sections", [])),
            }
        )
        meta = {
            "act_id": act_id,
            "code": raw["code"],
            "act_name": name_en,
            "act_name_bn": name_bn,
            "act_year": act_year,
            "category": raw["category"],
            "sections": 0,
            "with_en": 0,
            "with_bn": 0,
            "repealed": True,
            "repealed_reason": reason,
            "amended": amended,
            "source_url": source_url,
            "primary_source": "none",
        }
        return [], meta, dropped

    primary_name, primary_sections = choose_primary(raw)
    buckets = build_bilingual_buckets(raw, primary_name, primary_sections)

    bdlaws_footnotes = {
        int(f["n"]): f["text"]
        for f in (src.get("bdlaws") or {}).get("footnotes", [])
        if f.get("n") is not None
    }
    ukil_fn = ukil_footnotes(src.get("ukil"))
    fn_map = {**ukil_fn, **bdlaws_footnotes}

    titles = ukil_title_map(src.get("ukil"), set(buckets))
    sections: list[dict[str, Any]] = []
    for sec_no, bucket in buckets.items():
        text_en = bucket["text_en"]
        text_bn = bucket["text_bn"]
        if not text_en and not text_bn:
            continue
        if text_en:
            text_en = strip_leading_footnote_markers(text_en)
        if text_bn:
            text_bn = strip_leading_footnote_markers(text_bn)
        if section_is_repealed({"text": text_en or text_bn, "title": bucket["title"]}):
            dropped.append(
                {
                    "level": "section",
                    "act_id": act_id,
                    "act_name": name_en,
                    "section_no": sec_no,
                    "reason": "section text is Repealed/Omitted",
                }
            )
            continue

        title = bucket["title"]
        ukil_title = titles.get(sec_no, "")
        if ukil_title and ";" not in ukil_title:
            # Use the UKIL name only when the bdlaws title is empty, visibly
            # truncated, or a prefix with a short continuation.
            if (
                not title
                or title.endswith((",", "and", "or"))
                or (
                    ukil_title.lower().startswith(title.lower())
                    and 0 < len(ukil_title) - len(title) <= 60
                )
            ):
                title = ukil_title

        if text_en and text_bn:
            lang = "mixed"
        elif text_bn:
            lang = "bn"
        elif text_en:
            lang = "en"
        else:
            continue

        marks = sorted(bucket["footnote_markers"])
        section_footnotes = [
            fn_map[m] for m in marks if m in fn_map and fn_map[m]
        ]
        # de-duplicate while preserving order
        seen: set[str] = set()
        section_footnotes = [
            f for f in section_footnotes if not (f in seen or seen.add(f))
        ]

        sec_uid = (
            f"{raw['code']}_preamble"
            if sec_no == "preamble"
            else f"{raw['code']}_s{sec_no.lower()}"
        )
        sections.append(
            {
                "sec_uid": sec_uid,
                "act_id": act_id,
                "act_name": name_en,
                "section_no": sec_no,
                "title": title or None,
                "text_en": text_en,
                "text_bn": text_bn,
                "lang": lang,
                "repealed": False,
                "amended": amended,
                "footnotes": section_footnotes,
                # --- additive metadata -------------------------------------
                "act_name_bn": name_bn,
                "act_year": act_year,
                "category": raw["category"],
                "chapter": bucket["chapter"],
                "section_order": bucket["order"],
                "source_en": bucket["source_en"],
                "source_bn": bucket["source_bn"],
                "source_url": source_url,
            }
        )

    sections.sort(key=lambda s: (section_sort_key(s["section_no"]), s["section_order"]))

    meta = {
        "act_id": act_id,
        "code": raw["code"],
        "act_name": name_en,
        "act_name_bn": name_bn,
        "act_year": act_year,
        "category": raw["category"],
        "sections": len(sections),
        "with_en": sum(1 for s in sections if s["text_en"]),
        "with_bn": sum(1 for s in sections if s["text_bn"]),
        "repealed": False,
        "repealed_reason": None,
        "amended": amended,
        "source_url": source_url,
        "primary_source": primary_name,
        "bdlaws_available": bool(src.get("bdlaws")),
        "blad_available": bool(src.get("blad")),
        "ukil_available": bool(src.get("ukil")),
    }
    return sections, meta, dropped


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_report(
    act_meta: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    n_sections: int,
    sections: list[dict[str, Any]],
) -> str:
    included = [a for a in act_meta if not a["repealed"]]
    repealed_acts = [a for a in act_meta if a["repealed"]]
    en_only = [a for a in included if a["with_bn"] == 0 and a["with_en"] > 0]
    bn_only = [a for a in included if a["with_en"] == 0 and a["with_bn"] > 0]
    both = [a for a in included if a["with_en"] and a["with_bn"]]
    dropped_sections = [d for d in dropped if d["level"] == "section"]
    n_en = sum(1 for s in sections if s["text_en"])
    n_bn = sum(1 for s in sections if s["text_bn"])

    # numeric gaps that the official sources simply do not publish
    dropped_by_act: dict[int, set[str]] = defaultdict(set)
    for d in dropped_sections:
        if d.get("section_no"):
            dropped_by_act[int(d["act_id"])].add(str(d["section_no"]))
    present_by_act: dict[int, set[int]] = defaultdict(set)
    for s in sections:
        m = re.match(r"^(\d+)", str(s["section_no"]))
        if m:
            present_by_act[int(s["act_id"])].add(int(m.group(1)))
    gaps: list[tuple[str, list[int]]] = []
    for a in included:
        nums = present_by_act.get(a["act_id"], set())
        if not nums:
            continue
        top = max(nums)
        missing = [
            n
            for n in range(1, top + 1)
            if n not in nums and str(n) not in dropped_by_act.get(a["act_id"], set())
        ]
        if missing and len(missing) <= 40:
            gaps.append((a["act_name"], missing))

    lines: list[str] = []
    lines.append("# Stage 1 — Data collection report\n")
    lines.append(
        "Generated by `src/ingest.py` + `src/preprocess.py`. "
        "Canonical text: bdlaws (cached); fallback: BLAD; enrichment: UKIL-DB-EN.\n"
    )
    lines.append("## Headline numbers\n")
    lines.append(f"- Acts in whitelist: **{len(act_meta)}**")
    lines.append(f"- Acts included: **{len(included)}**")
    lines.append(f"- Acts dropped as repealed: **{len(repealed_acts)}**")
    lines.append(f"- Sections in `sections.jsonl`: **{n_sections:,}**")
    lines.append(f"- Sections with English text: **{n_en:,}**")
    lines.append(f"- Sections with Bangla text: **{n_bn:,}**")
    lines.append(f"- Repealed/omitted sections dropped: **{len(dropped_sections):,}**")
    lines.append(f"- Acts with English text: **{sum(1 for a in included if a['with_en'])}**")
    lines.append(f"- Acts with Bangla text: **{sum(1 for a in included if a['with_bn'])}**")
    lines.append(f"- Fully bilingual acts: **{len(both)}**")
    lines.append("")

    lines.append("## Output schema\n")
    lines.append(
        "One JSON object per section in `data/clean/sections.jsonl`. "
        "`sec_uid` is the stable join key across languages "
        "(`{code}_s{section_no}`, e.g. `pc1860_s302`); the preamble is "
        "`{code}_preamble`.\n"
    )
    lines.append("| Field | Meaning |")
    lines.append("|---|---|")
    lines.append("| `sec_uid` | Stable section id, language-independent |")
    lines.append("| `act_id` | bdlaws/BLAD/UKIL act id |")
    lines.append("| `act_name` | Official English act name |")
    lines.append("| `section_no` | Section number (`302`, `171A`, `preamble`) |")
    lines.append("| `title` | Marginal title of the section |")
    lines.append("| `text_en` | Official English text, or `null` |")
    lines.append("| `text_bn` | Official Bangla text, or `null` |")
    lines.append("| `lang` | `en`, `bn`, or `mixed` (both texts present) |")
    lines.append("| `repealed` | Always `false` here; repealed sections are dropped |")
    lines.append("| `amended` | Latest amendment date/year found in the footnotes, or `null` |")
    lines.append("| `footnotes` | Amendment/source notes attached to this section |")
    lines.append("")
    lines.append(
        "Extra columns (`act_name_bn`, `act_year`, `category`, `chapter`, "
        "`section_order`, `source_en`, `source_bn`, `source_url`) are added for "
        "filtering and provenance.\n"
    )

    lines.append("## Language coverage\n")
    lines.append(
        "Bangla and English are joined per section through `sec_uid`. "
        "The supplied corpora are monolingual per act, so most records have one "
        "language only. This is reported explicitly; no text was machine-translated.\n"
    )
    lines.append("| Act ID | Code | Act | Sections | EN | BN | Amended | Primary |")
    lines.append("|---:|---|---|---:|---:|---:|---|---|")
    for a in sorted(included, key=lambda x: x["act_id"]):
        lines.append(
            f"| {a['act_id']} | {a['code']} | {a['act_name']} | {a['sections']} | "
            f"{a['with_en']} | {a['with_bn']} | {a['amended'] or '—'} | {a['primary_source']} |"
        )
    lines.append("")

    lines.append("## English-only acts (Bangla gap to solve at retrieval)\n")
    for a in sorted(en_only, key=lambda x: x["act_id"]):
        lines.append(f"- {a['act_name']} (`{a['code']}`, {a['sections']} sections)")
    lines.append("")
    lines.append("## Bangla-only acts\n")
    for a in sorted(bn_only, key=lambda x: x["act_id"]):
        lines.append(f"- {a['act_name']} (`{a['code']}`, {a['sections']} sections)")
    lines.append("")

    lines.append("## Repealed acts excluded\n")
    for a in sorted(repealed_acts, key=lambda x: x["act_id"]):
        lines.append(f"- {a['act_name']} — {a['repealed_reason']}")
    lines.append("")

    lines.append("## Numbering gaps in the official source\n")
    if gaps:
        lines.append(
            "These section numbers are absent from every supplied source "
            "(bdlaws, BLAD and UKIL all skip them); they are omitted provisions, "
            "not ingestion failures. Repealed sections already dropped are "
            "excluded from this list.\n"
        )
        for name, missing in gaps:
            shown = ", ".join(str(x) for x in missing)
            lines.append(f"- **{name}**: {shown}")
    else:
        lines.append("No unexplained gaps.")
    lines.append("")

    lines.append("## Method notes\n")
    lines.append(
        "1. **Section numbers** are recovered from the text in both scripts "
        "(`302.`, `১৭১ক।`), with leading footnote markers stripped. Unnumbered "
        "rows are merged into the preceding section; preamble rows are kept as "
        "`section_no=\"preamble\"`."
    )
    lines.append(
        "2. **Repealed sections** are dropped when their whole text is a "
        "Repealed/Omitted note (`[Repealed by ...]`, `[***]`, `রহিত`). Partially "
        "omitted sub-sections are kept, as the rest of the section remains in force."
    )
    lines.append(
        "3. **Amendment date** is the latest date/year found in the act's own "
        "footnotes (and, for BLAD-only acts, its section text), not the enactment "
        "year. It is an indicator, not a legal opinion."
    )
    lines.append(
        "4. **Footnotes** are attached per section using the inline footnote "
        "markers (bdlaws `<sup>` numbers; UKIL numbering as fallback)."
    )
    lines.append(
        "5. **Sources of record**: bdlaws is current to 2026 and also supplied "
        "the repealed flags; BLAD (2025 snapshot) is the offline fallback; UKIL "
        "supplies English section names and footnotes. Pages are cached in "
        "`data/raw/bdlaws/html`, so re-runs are offline and reproducible."
    )
    lines.append("")
    lines.append("## Coverage gaps\n")
    lines.append(
        "- Official Bangla text of the colonial codes (Penal Code 1860, CrPC 1898, "
        "Evidence Act 1872, Police Act 1861, Arms Act 1878, etc.) is absent from "
        "bdlaws/BLAD/UKIL. Their `text_bn` is null by design.\n"
        "- Official English text of post-1972 Bangla statutes is likewise absent; "
        "their `text_en` is null. Bangla queries still reach English sections via "
        "cross-lingual retrieval at Stage 2.\n"
        "- This is a property of the supplied sources, not of the pipeline: a "
        "true parallel English–Bangla corpus is not available for these acts.\n"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated act ids")
    args = parser.parse_args(argv)

    only = {int(x) for x in re.findall(r"\d+", args.only)} if args.only else None
    paths = sorted(C.RAW_ACTS.glob("act-*.json"))
    if not paths:
        print("no raw act records; run `python -m src.ingest` first")
        return 1

    all_sections: list[dict[str, Any]] = []
    act_meta: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if only and raw["act_id"] not in only:
            continue
        sections, meta, act_dropped = build_act(raw)
        all_sections.extend(sections)
        act_meta.append(meta)
        dropped.extend(act_dropped)
        print(
            f"  {raw['act_id']:4d} {raw['code']:22s} "
            f"sections={meta['sections']:4d} en={meta['with_en']:4d} bn={meta['with_bn']:4d} "
            f"{'REPEALED' if meta['repealed'] else ''}"
        )

    # ---- integrity checks -------------------------------------------------
    uid_counts = Counter(s["sec_uid"] for s in all_sections)
    duplicates = {k: v for k, v in uid_counts.items() if v > 1}
    if duplicates:
        raise SystemExit(f"duplicate sec_uid values: {list(duplicates.items())[:10]}")

    required = {
        "sec_uid",
        "act_id",
        "act_name",
        "section_no",
        "title",
        "text_en",
        "text_bn",
        "lang",
        "repealed",
        "amended",
        "footnotes",
    }
    for sec in all_sections:
        missing = required - sec.keys()
        if missing:
            raise SystemExit(f"{sec.get('sec_uid')} missing keys: {missing}")
        if sec["lang"] not in {"en", "bn", "mixed"}:
            raise SystemExit(f"bad lang {sec['lang']} in {sec['sec_uid']}")

    n_sections = write_jsonl(C.DATA_CLEAN / "sections.jsonl", all_sections)
    write_jsonl(C.DATA_CLEAN / "acts.jsonl", act_meta)
    write_jsonl(C.DATA_CLEAN / "dropped.jsonl", dropped)

    C.DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    with (C.DATA_CLEAN / "coverage.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "act_id",
                "code",
                "act_name",
                "act_year",
                "category",
                "sections",
                "with_en",
                "with_bn",
                "repealed",
                "amended",
                "primary_source",
                "source_url",
            ]
        )
        for a in act_meta:
            writer.writerow(
                [
                    a["act_id"],
                    a["code"],
                    a["act_name"],
                    a["act_year"],
                    a["category"],
                    a["sections"],
                    a["with_en"],
                    a["with_bn"],
                    a["repealed"],
                    a["amended"] or "",
                    a["primary_source"],
                    a["source_url"],
                ]
            )

    C.REPORT.mkdir(parents=True, exist_ok=True)
    report = build_report(act_meta, dropped, n_sections, all_sections)
    (C.REPORT / "stage1_report.md").write_text(report, encoding="utf-8")

    print(f"\n[done] {n_sections} sections -> {C.DATA_CLEAN / 'sections.jsonl'}")
    print(f"[done] report -> {C.REPORT / 'stage1_report.md'}")

    # spot checks for the core criminal provisions
    spot = {s["sec_uid"]: s for s in all_sections}
    for uid in ("pc1860_s302", "crpc1898_s154", "ev1872_s3", "wc2000_s9", "narcotics2018_s36"):
        sec = spot.get(uid)
        if sec:
            print(
                f"[check] {uid}: {sec['lang']:5s} | {(sec['title'] or '')[:50]} | "
                f"{(sec['text_en'] or sec['text_bn'] or '')[:70]}"
            )
        else:
            print(f"[check] {uid}: MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
