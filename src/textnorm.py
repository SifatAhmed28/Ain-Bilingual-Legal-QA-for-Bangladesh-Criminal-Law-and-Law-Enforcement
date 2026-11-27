"""Stage 2 text cleaning, Unicode normalisation, language ID and segmentation.

The functions here are deliberately stateless and side-effect free so that the
same code can be applied to corpus sections, user queries and unit tests.  Every
function returns the transformed text together with a small counter of what it
changed; the pipeline aggregates those counters for the Stage 2 report.

Design notes
------------
* Cleaning removes *presentation* artefacts only (HTML, page furniture,
  footnote markers, duplicated section numbers).  Amendment brackets such as
  ``[জুডিসিয়াল ম্যাজিস্ট্রেট]`` are legal content and are kept.
* Bangla nukta letters (ড় ঢ় য়) are composition-excluded in Unicode, so NFC
  decomposes them.  We therefore run NFC and then re-compose them explicitly to
  U+09DC/U+09DD/U+09DF, which is the form used by 99% of the source text and by
  Bangla keyboards.  Queries must go through the same function.
* Language ID is a script check, not a statistical language model.  It is cheap,
  deterministic and adequate for a two-script corpus; romanised Bangla is
  reported as a known limitation.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# character classes
# ---------------------------------------------------------------------------

# Bangla letters (base characters), i.e. the U+0980-09FF block minus digits,
# vowel signs and punctuation.  Used for script counting and language ID.
BN_LETTER_RE = re.compile(r"[\u0985-\u09B9\u09CE\u09DC-\u09DF\u09F0-\u09F1]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
BN_DIGIT_RE = re.compile(r"[০-৯]")
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

BN_NUKTA_PRECOMPOSED = ("\u09DC", "\u09DD", "\u09DF")  # ড় ঢ় য়
BN_NUKTA_DECOMPOSED = ("\u09A1\u09BC", "\u09A2\u09BC", "\u09AF\u09BC")
BN_RECOMPOSE = {
    "\u09A1\u09BC": "\u09DC",
    "\u09A2\u09BC": "\u09DD",
    "\u09AF\u09BC": "\u09DF",
}
# Vowel signs that are canonically decomposable: ো (ে + া) and ৌ (ে + ৗ).
BN_SPLIT_VOWELS = ("\u09C7\u09BE", "\u09C7\u09D7")

BN_SECTION_SUFFIX = {
    "ক": "A",
    "খ": "B",
    "গ": "C",
    "ঘ": "D",
    "ঙ": "E",
    "চ": "F",
    "ছ": "G",
    "জ": "H",
}

# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<[^>]{1,300}>")
ENTITY_RE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#x[0-9A-Fa-f]{1,6});")
ZW_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
HYPHEN_JOIN_RE = re.compile(r"(\w)-\n(\w)")
PAGE_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:page[ \t]*)?\d{1,4}(?:[ \t]*(?:of|/)[ \t]*\d{1,4})?[ \t]*$"
)
DASH_PAGE_RE = re.compile(r"(?m)^[ \t]*[-\u2013\u2014][ \t]*\d{1,4}[ \t]*[-\u2013\u2014][ \t]*$")
OMITTED_RE = re.compile(r"\[\s*(?:\*\s*){2,}\]")
LEADING_FN_RE = re.compile(r"^\s*\[\s*(?:\d{1,3}|[০-৯]{1,3})\s*\]\s*")
GLUED_FN_RE = re.compile(r"(?<![\d০-৯])(?:\d{1,3}|[০-৯]{1,3})\[")
SUPERSCRIPT_RE = re.compile(r"[\u2070\u00B9\u00B2\u00B3\u2074-\u2079]+")
SEC_NUM_RE = re.compile(
    r"^\s*([0-9০-৯]{1,3}[A-Za-z\u0995-\u09B9]{0,3})"
    r"(?:\s*(?:-|and|to|এবং|থেকে)\s*[0-9০-৯]{1,3}[A-Za-z\u0995-\u09B9]{0,3})?"
    r"\s*[.\u0964\u09F7]\s*"
)

RUNNING_HEADS = (
    "গণপ্রজাতন্ত্রী বাংলাদেশ সরকার",
    "Government of the People's Republic of Bangladesh",
)


def _canon_section_number(raw: str) -> str:
    out = []
    for ch in raw:
        if ch in BN_SECTION_SUFFIX:
            out.append(BN_SECTION_SUFFIX[ch])
        else:
            out.append(ch)
    return "".join(out).translate(BN_DIGITS).upper()


def strip_section_number(text: str, section_no: str | None) -> tuple[str, bool]:
    """Drop the leading ``302.`` / ``২১।`` that duplicates the metadata field.

    Only strips when the parsed number equals ``section_no``; a leading footnote
    digit that happens to look like a number is therefore left for the footnote
    rule.  A following ``(1)`` is a sub-section marker and is preserved.
    """
    if not text or not section_no or section_no == "preamble":
        return text, False
    m = SEC_NUM_RE.match(text)
    if not m:
        return text, False
    if _canon_section_number(m.group(1)) != str(section_no).upper():
        return text, False
    return text[m.end() :], True


def strip_running_headers(text: str, extra: Iterable[str] = ()) -> tuple[str, int]:
    """Remove whole lines that are running heads (act name / government line).

    Whole-line matching only: an act name mentioned inside a sentence survives.
    """
    targets = {_squash(x) for x in (*RUNNING_HEADS, *extra) if x}
    if not targets:
        return text, 0
    out: list[str] = []
    removed = 0
    for line in text.split("\n"):
        if _squash(line) in targets:
            removed += 1
            continue
        out.append(line)
    return "\n".join(out), removed


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def clean_text(
    text: str | None,
    section_no: str | None = None,
    running_heads: Iterable[str] = (),
) -> tuple[str, Counter]:
    """Remove presentation artefacts and return ``(text, counts)``."""
    counts: Counter = Counter()
    if not text:
        return "", counts

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1. HTML leftovers -----------------------------------------------------
    n_ent = len(ENTITY_RE.findall(text))
    if n_ent:
        counts["html_entities"] += n_ent
        text = html.unescape(text)
    n_tag = len(TAG_RE.findall(text))
    if n_tag:
        counts["html_tags"] += n_tag
        text = TAG_RE.sub(" ", text)

    # 2. Invisible characters ----------------------------------------------
    n_zw = len(ZW_RE.findall(text))
    if n_zw:
        counts["zero_width"] += n_zw
        text = ZW_RE.sub("", text)

    # 3. Line-level furniture ----------------------------------------------
    text, n = HYPHEN_JOIN_RE.subn(r"\1\2", text)
    counts["line_hyphenation"] += n
    text, n = PAGE_LINE_RE.subn("", text)
    counts["page_number_lines"] += n
    text, n = DASH_PAGE_RE.subn("", text)
    counts["page_number_dashes"] += n
    text, n = strip_running_headers(text, running_heads)
    counts["running_headers"] += n

    # 4. Footnote and amendment markers ------------------------------------
    text, stripped = strip_section_number(text, section_no)
    counts["section_numbers"] += int(stripped)
    text, n = LEADING_FN_RE.subn("", text)
    counts["footnote_markers"] += n

    def _strip_glued(m: re.Match[str]) -> str:
        before = text[m.start() - 1] if m.start() else " "
        # "to409[transportation]" -> "to [transportation]" keeps the word and
        # the amendment bracket separate tokens.
        return "[" if before.isspace() else " ["

    text, n = GLUED_FN_RE.subn(_strip_glued, text)
    counts["footnote_markers"] += n
    text, n = SUPERSCRIPT_RE.subn("", text)
    counts["footnote_markers"] += n
    # "[* * *]" marks an omitted passage; the asterisks carry no retrieval value.
    text, n = OMITTED_RE.subn("", text)
    counts["omitted_passages"] += n

    # 5. Whitespace ---------------------------------------------------------
    counts["double_spaces"] += len(re.findall(r" {2,}", text))
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()

    return text, counts


# ---------------------------------------------------------------------------
# Unicode normalisation
# ---------------------------------------------------------------------------


def normalize_text(text: str | None) -> tuple[str, Counter]:
    """NFC + Bangla-specific canonicalisation + digit unification.

    Returns ``(text, counts)``.  The counts describe the *input*, which is what
    the report needs; they are not a diff against the output.
    """
    counts: Counter = Counter()
    if not text:
        return "", counts

    counts["nukta_precomposed"] = sum(text.count(ch) for ch in BN_NUKTA_PRECOMPOSED)
    counts["nukta_decomposed"] = sum(text.count(seq) for seq in BN_NUKTA_DECOMPOSED)
    counts["split_vowel_signs"] = sum(text.count(seq) for seq in BN_SPLIT_VOWELS)
    counts["bn_digits"] = len(BN_DIGIT_RE.findall(text))
    counts["danda_variants"] = text.count("\u09F7")
    counts["zero_width"] = len(ZW_RE.findall(text))

    text = unicodedata.normalize("NFC", text)

    # NFC decomposes the composition-excluded nukta letters; put them back into
    # the pre-composed form that the corpus and Bangla keyboards use.
    recomposed = 0
    for seq, composed in BN_RECOMPOSE.items():
        n = text.count(seq)
        if n:
            recomposed += n
            text = text.replace(seq, composed)
    counts["nukta_recomposed"] = recomposed

    # Bengali danda U+09F7 -> Devanagari danda U+0964 (the terminator used by
    # the rest of the corpus and by the sentence splitter).
    text = text.replace("\u09F7", "\u0964")
    text = text.translate(BN_DIGITS)
    return text, counts


def is_canonical(text: str) -> bool:
    """True when ``normalize_text`` is idempotent on ``text``."""
    return normalize_text(text)[0] == text


# ---------------------------------------------------------------------------
# language identification
# ---------------------------------------------------------------------------

BN_PURITY_THRESHOLD = 0.85


def script_counts(text: str) -> tuple[int, int]:
    """Return ``(bangla_letters, latin_letters)``."""
    return len(BN_LETTER_RE.findall(text)), len(LATIN_LETTER_RE.findall(text))


def detect_lang(text: str) -> tuple[str, float]:
    """Label a segment ``bn``, ``en`` or ``mixed`` and return the Bangla share.

    A segment with no letters (``"(1)"``, ``"1973"``) is ``neutral``; callers
    inherit the parent segment's language in that case.
    """
    bn, latin = script_counts(text)
    total = bn + latin
    if total == 0:
        return "neutral", 0.0
    share = bn / total
    if share >= BN_PURITY_THRESHOLD:
        return "bn", share
    if share <= 1.0 - BN_PURITY_THRESHOLD:
        return "en", share
    return "mixed", share


# ---------------------------------------------------------------------------
# sentence segmentation
# ---------------------------------------------------------------------------

# Abbreviations whose full stop must not end a sentence.  Kept small and
# legal-domain specific; the dotted-acronym rule below catches U.S., i.e., P.C.
ABBREVIATIONS = (
    "No",
    "Sec",
    "ss",
    "Art",
    "Cl",
    "etc",
    "viz",
    "vs",
    "v",
    "Hon",
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "St",
    "Co",
    "Ltd",
    "Govt",
    "Dist",
    "Addl",
    "Asstt",
    "Suptd",
    "Insp",
    "para",
    "Sch",
    "Pt",
    "Fig",
    "Vol",
    "Ed",
    "pp",
    "p",
    "Rs",
    "Tk",
)
ABBREV_RE = re.compile(r"\b(?:" + "|".join(ABBREVIATIONS) + r")\.", re.I)
DOTTED_RE = re.compile(r"(?:[A-Za-z]\.){2,}")
PROTECT = "\u0001"

_BN_START = "\u0985-\u09B9"
_STARTERS = set("\"'“‘)]")


def _protect_abbreviations(text: str) -> str:
    """Replace non-final full stops with a placeholder, preserving length."""

    def repl(m: re.Match[str]) -> str:
        return m.group(0).replace(".", PROTECT)

    text = DOTTED_RE.sub(repl, text)
    text = ABBREV_RE.sub(repl, text)
    # Decimal numbers: 3.5, 0.25
    text = re.sub(r"(?<=\d)\.(?=\d)", PROTECT, text)
    return text


def _next_significant(text: str, i: int) -> str:
    while i < len(text) and text[i].isspace():
        i += 1
    while i < len(text) and text[i] in "\"'“‘":
        i += 1
    return text[i] if i < len(text) else ""


CLOSERS = "])}\"'”’"


def _skip_closers(text: str, i: int) -> int:
    while i < len(text) and text[i] in CLOSERS:
        i += 1
    return i


def _is_sentence_end(text: str, i: int, j: int) -> bool:
    ch = text[i]
    if ch in "\u0964\u09F7":
        return True
    # Look past closing brackets/quotes that hug the terminator: in "Go?\"" the
    # question mark still ends the sentence.
    k = _skip_closers(text, j)
    if ch in "!?":
        return k >= len(text) or text[k].isspace()
    if ch == ".":
        if k < len(text) and not text[k].isspace():
            return False
        nxt = _next_significant(text, k)
        if not nxt:
            return True
        return (
            nxt.isupper()
            or nxt.isdigit()
            or ("\u0985" <= nxt <= "\u09B9")
            or nxt in _STARTERS
        )
    return False


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into ``(sentence, start, end)`` spans.

    Handles the Bangla danda ``।``, the variant ``৷``, ``?``/``!`` and English
    full stops with an abbreviation guard.  Sentence-final punctuation is kept
    in the sentence so that joining sentences reproduces the section text.
    """
    if not text:
        return []
    protected = _protect_abbreviations(text)
    spans: list[tuple[str, int, int]] = []
    start = 0
    for m in re.finditer(r"[\u0964\u09F7!?.]+", protected):
        i, j = m.start(), m.end()
        # "?!" / "!!!" are one terminator run
        if not _is_sentence_end(protected, i, j):
            continue
        # A run of terminators ends at its last character, but a closing
        # bracket that hugs it belongs to the same sentence: "[... হয়।]"
        # must not leave a lone "]" as the next sentence.
        end = _skip_closers(text, j)
        piece = text[start:end].strip()
        if piece:
            lead = len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((piece, start + lead, end))
        start = end
    tail = text[start:].strip()
    if tail:
        lead = len(text[start:]) - len(text[start:].lstrip())
        spans.append((tail, start + lead, len(text)))

    # A span with no letters or digits ("]", "***") is an artefact of a
    # bracket or footnote; fold it into the neighbouring sentence.
    merged: list[tuple[str, int, int]] = []
    for span in spans:
        if not CLAUSE_CONTENT_RE.search(span[0]) and merged:
            prev = merged[-1]
            merged[-1] = (text[prev[1] : span[2]].strip(), prev[1], span[2])
        else:
            merged.append(span)
    while len(merged) > 1 and not CLAUSE_CONTENT_RE.search(merged[0][0]):
        first = merged.pop(0)
        nxt = merged.pop(0)
        merged.insert(0, (text[first[1] : nxt[2]].strip(), first[1], nxt[2]))
    return merged


# ---------------------------------------------------------------------------
# clause segmentation
# ---------------------------------------------------------------------------

CLAUSE_MARKER_RE = re.compile(
    r"(?:(?<=[\s\[\(\"'“‘—–-])|^)\((?P<marker>\d{1,2}|[০-৯]{1,2}|[A-Za-z]|[ক-হ])\)\s+"
)
CLAUSE_CONTENT_RE = re.compile(r"[A-Za-z0-9\u0985-\u09B9]")
PROVISO_RE = re.compile(
    r"(?=(?:Provided (?:further )?that|তবে (?:আরও )?শর্ত থাকে যে|Explanation|ব্যাখ্যা))"
)


def _clause_type(marker: str) -> str:
    if not marker:
        return "plain"
    if marker[0].isdigit() or "\u09E6" <= marker[0] <= "\u09EF":
        return "sub-section"
    return "paragraph"


def split_clauses(text: str) -> list[dict[str, object]]:
    """Split a sentence/section into sub-section and proviso clauses.

    The marker itself is retained at the front of ``text`` so that a chunk can
    never lose ``(2)`` or ``(খ)`` when it is cut out of the middle of a section.
    """
    if not text:
        return []
    marks = list(CLAUSE_MARKER_RE.finditer(text))
    # If the marker sits inside an amendment bracket, "[(2) ...]", keep the
    # opening bracket with the clause so the bracket pair is not left dangling.
    starts = [
        m.start() - 1 if m.start() > 0 and text[m.start() - 1] == "[" else m.start()
        for m in marks
    ]
    segments: list[tuple[int, int, str]] = []
    if not marks:
        segments.append((0, len(text), ""))
    else:
        if text[: starts[0]].strip():
            segments.append((0, starts[0], ""))
        for k, m in enumerate(marks):
            end = starts[k + 1] if k + 1 < len(starts) else len(text)
            segments.append((starts[k], end, m.group("marker")))

    out: list[dict[str, object]] = []
    for start, end, marker in segments:
        raw = text[start:end]
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        if not body or not CLAUSE_CONTENT_RE.search(body):
            continue
        # Split provisos / explanations out of the clause.
        pieces = [p for p in PROVISO_RE.split(body) if p.strip()]
        for p in pieces:
            p_start = body.find(p)
            p_text = p.strip()
            p_lead = len(p) - len(p.lstrip())
            lower = p_text.lower()
            if lower.startswith("provided") or p_text.startswith("তবে"):
                ctype = "proviso"
            elif lower.startswith("explanation") or p_text.startswith("ব্যাখ্যা"):
                ctype = "explanation"
            else:
                ctype = _clause_type(marker)
            out.append(
                {
                    "marker": marker,
                    "type": ctype,
                    "text": p_text,
                    "start": start + lead + p_start + p_lead,
                    "end": start + lead + p_start + p_lead + len(p_text),
                }
            )
    return out


def split_all(text: str) -> list[dict[str, object]]:
    """Sentence + clause segmentation used by the pipeline."""
    rows: list[dict[str, object]] = []
    for s_idx, (sent, s_start, s_end) in enumerate(split_sentences(text)):
        lang, share = detect_lang(sent)
        clauses = split_clauses(sent)
        for clause in clauses:
            # split_clauses works inside the sentence; lift the offsets back to
            # the section so segment spans are directly usable for highlighting.
            clause["start"] = int(clause["start"]) + s_start
            clause["end"] = int(clause["end"]) + s_start
        rows.append(
            {
                "index": s_idx,
                "text": sent,
                "start": s_start,
                "end": s_end,
                "lang": lang,
                "bn_share": round(share, 4),
                "clauses": clauses,
            }
        )
    return rows
