"""Stage 2: core NLP preprocessing for the bilingual legal corpus.

Pipeline
--------
``data/clean/sections.jsonl``
    -> clean (HTML, page furniture, footnote markers, duplicate section no.)
    -> normalise (NFC + Bangla recomposition + digit unification)
    -> language ID (script ratio per section / sentence / clause)
    -> sentence and clause segmentation (markers retained)
    -> tokenise (whitespace baseline + embedding-model subword tokenizer)
    -> BM25 view (stopwords + light stemming; never used for dense)
    -> chunk (one section = one chunk, <=400 subword tokens, 50 overlap)

Outputs
-------
``data/clean/sections_norm.jsonl``
    One row per section: canonical text, BM25 text, language, counts.
``data/clean/segments.jsonl``
    Sentence and clause spans with stable ids.
``data/clean/chunks.jsonl``
    Retrieval units; every row keeps ``sec_uid``.
``results/stage2_token_stats.csv``
    Per-section token counts for the report and for Stage 3 budgeting.
``results/stage2_summary.json``
    Aggregate numbers used by ``report/stage2_report.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config as C
from . import textnorm as T
from .bn_nlp import (
    BN_STOPWORDS,
    EN_STOPWORDS,
    BaseTokenizer,
    bm25_text,
    load_tokenizer,
    stem_bn,
    stem_en,
    whitespace_tokens,
    word_tokens,
)
from .lexicon import LegalTerm, expand_query, load_lexicon, validate_statutes

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STOPWORDS = BN_STOPWORDS | EN_STOPWORDS
TOKEN_MARGIN = 4  # tokenizer merges across the header/body boundary


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def make_header(sec: dict[str, Any], lang: str) -> str:
    """Self-describing first line so a retrieved chunk is citable on its own."""
    section_no = str(sec.get("section_no") or "")
    if lang == "bn":
        act = sec.get("act_name_bn") or sec.get("act_name") or ""
        head = f"{act}, প্রস্তাবনা" if section_no == "preamble" else f"{act}, ধারা {section_no}"
    else:
        act = sec.get("act_name") or ""
        head = f"{act} — Preamble" if section_no == "preamble" else f"{act}, s. {section_no}"
    title = (sec.get("title") or "").strip()
    if title:
        if len(title) > 120:
            title = title[:117].rstrip() + "..."
        head += f" — {title}"
    return head


def _split_long_unit(text: str, budget: int, tok: BaseTokenizer) -> list[str]:
    """Word-level split for a single sentence longer than the budget."""
    words = text.split()
    if len(words) <= 1:
        return [text]
    counts = [max(1, n) for n in tok.count_batch(words)]
    pieces: list[str] = []
    i = 0
    while i < len(words):
        j = i
        total = 0
        while j < len(words) and (total + counts[j] <= budget or j == i):
            total += counts[j]
            j += 1
        pieces.append(" ".join(words[i:j]))
        i = j
    return pieces


def _tail_start(text: str, max_tokens: int, tok: BaseTokenizer) -> int:
    """Character offset where the last <=``max_tokens`` tokens begin."""
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return len(text)
    counts = [max(1, n) for n in tok.count_batch([m.group() for m in matches])]
    total = 0
    k = len(matches)
    while k - 1 >= 0 and total + counts[k - 1] <= max_tokens:
        k -= 1
        total += counts[k]
    return matches[k].start() if k < len(matches) else len(text)


def chunk_section(
    sec: dict[str, Any],
    lang: str,
    sentences: Sequence[str],
    sentence_offsets: Sequence[tuple[int, int]],
    sentence_tokens: Sequence[int],
    section_text: str,
    tok: BaseTokenizer,
    max_tokens: int = C.CHUNK_MAX_TOKENS,
    overlap: int = C.CHUNK_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """Greedy sentence-boundary chunking with a guaranteed token overlap.

    Sentences are the preferred cut points, but the overlap window is measured
    in subword tokens and may start inside a sentence.  That is intentional: a
    400-token legal sentence cannot be overlapped by whole sentences without
    duplicating the whole chunk.

    Returns chunk dicts without ``chunk_index``/``n_chunks`` (filled by caller).
    """
    header = make_header(sec, lang)
    header_tokens = tok.count(header)
    budget = max(60, max_tokens - header_tokens - TOKEN_MARGIN)
    # Reserve room for the overlap so the first new sentence always fits.
    unit_budget = max(30, budget - overlap)

    # Build units: normally one sentence, but split pathological sentences.
    # Each unit carries its exact span in the section so chunk offsets stay
    # precise even when a sentence is cut mid-way.
    if len(sentence_offsets) != len(sentences):
        sentence_offsets = [(0, 0)] * len(sentences)
    units: list[tuple[str, int, int, int]] = []
    for idx, sent in enumerate(sentences):
        s_start, s_end = sentence_offsets[idx]
        if sentence_tokens[idx] <= unit_budget:
            units.append((sent, idx, s_start, s_end))
        else:
            cursor = s_start
            for piece in _split_long_unit(sent, unit_budget, tok):
                pos = section_text.find(piece, cursor)
                if pos == -1:
                    pos = cursor
                units.append((piece, idx, pos, pos + len(piece)))
                cursor = pos + len(piece)
    if not units:
        units = [("", 0, 0, 0)]

    unit_tokens = tok.count_batch([u[0] for u in units])
    raw: list[dict[str, Any]] = []
    i = 0
    overlap_start: int | None = None
    prev_char_end = 0
    prev_body = ""
    while i < len(units):
        if overlap_start is not None:
            char_start = overlap_start
            base = tok.count(section_text[char_start:prev_char_end])
        else:
            char_start = units[i][2]
            base = 0
        j = i
        total = base
        while j < len(units) and (total + unit_tokens[j] <= budget or j == i):
            total += unit_tokens[j]
            j += 1
        char_end = units[j - 1][3]
        # Slice the section instead of re-joining units: this keeps the body
        # byte-identical to the source, including the common "।(2)" spelling
        # where two sentences have no space between them.
        body = section_text[char_start:char_end]
        text = f"{header}\n{body}" if body else header
        overlap_text = section_text[char_start:prev_char_end] if overlap_start is not None else ""

        raw.append(
            {
                "text": text,
                "body": body,
                "n_tokens": tok.count(text),
                "sentence_start": units[i][1],
                "sentence_end": units[j - 1][1],
                "overlap_tokens": base,
                "overlap_ok": (not overlap_text) or prev_body.endswith(overlap_text),
                "char_start": char_start,
                "char_end": char_end,
            }
        )
        if j >= len(units):
            break
        prev_char_end = char_end
        prev_body = body
        tail = _tail_start(body, overlap, tok)
        overlap_start = char_start + tail if tail < len(body) else None
        i = j
    return raw


# ---------------------------------------------------------------------------
# per-section processing
# ---------------------------------------------------------------------------


def process_section(
    sec: dict[str, Any],
    tok: BaseTokenizer,
    act_code: str,
    max_tokens: int = C.CHUNK_MAX_TOKENS,
    overlap: int = C.CHUNK_OVERLAP_TOKENS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return ``(section_row, segments, chunks, per_section_stats)``."""
    lang = "bn" if sec.get("text_bn") else "en"
    raw = sec.get("text_bn") or sec.get("text_en") or ""

    cleaned, clean_counts = T.clean_text(
        raw,
        section_no=sec.get("section_no"),
        running_heads=(sec.get("act_name"), sec.get("act_name_bn")),
    )
    normalised, norm_counts = T.normalize_text(cleaned)
    script_lang, share = T.detect_lang(normalised)
    if script_lang == "neutral":
        script_lang = lang

    # Segmentation --------------------------------------------------------
    sentences = T.split_all(normalised)
    sentence_texts = [s["text"] for s in sentences]
    sentence_tokens = tok.count_batch(sentence_texts) if sentence_texts else []
    sentence_offsets = [(s["start"], s["end"]) for s in sentences]

    segments: list[dict[str, Any]] = []
    clause_total = 0
    clause_tokens: list[int] = []
    clause_texts: list[str] = []
    for s in sentences:
        s_idx = s["index"]
        s_lang = s["lang"]
        inherited = s_lang == "neutral"
        if inherited:
            s_lang = script_lang
        segments.append(
            {
                "seg_id": f"{sec['sec_uid']}:s{s_idx:03d}",
                "sec_uid": sec["sec_uid"],
                "kind": "sentence",
                "parent_id": None,
                "index": s_idx,
                "marker": "",
                "clause_type": "",
                "lang": s_lang,
                "lang_inherited": inherited,
                "text": s["text"],
                "char_start": s["start"],
                "char_end": s["end"],
                "n_tokens": sentence_tokens[s_idx] if s_idx < len(sentence_tokens) else 0,
            }
        )
        for c_idx, clause in enumerate(s["clauses"]):
            clause_total += 1
            clause_texts.append(str(clause["text"]))
            segments.append(
                {
                    "seg_id": f"{sec['sec_uid']}:c{clause_total - 1:03d}",
                    "sec_uid": sec["sec_uid"],
                    "kind": "clause",
                    "parent_id": f"{sec['sec_uid']}:s{s_idx:03d}",
                    "index": c_idx,
                    "marker": clause["marker"],
                    "clause_type": clause["type"],
                    "lang": s_lang,
                    "lang_inherited": inherited,
                    "text": clause["text"],
                    "char_start": clause["start"],
                    "char_end": clause["end"],
                    "n_tokens": 0,
                }
            )
    if clause_texts:
        clause_tokens = tok.count_batch(clause_texts)
        for seg, n in zip((s for s in segments if s["kind"] == "clause"), clause_tokens):
            seg["n_tokens"] = n

    # Chunks ---------------------------------------------------------------
    raw_chunks = chunk_section(
        sec,
        script_lang,
        sentence_texts,
        sentence_offsets,
        sentence_tokens,
        normalised,
        tok,
        max_tokens,
        overlap,
    )
    chunks: list[dict[str, Any]] = []
    for n, row in enumerate(raw_chunks, start=1):
        chunk_id = f"{sec['sec_uid']}#c{n:02d}"
        text = row["text"]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "sec_uid": sec["sec_uid"],
                "act_id": sec["act_id"],
                "act_code": act_code,
                "act_name": sec.get("act_name"),
                "section_no": sec.get("section_no"),
                "title": sec.get("title"),
                "lang": script_lang,
                "chunk_index": n,
                "n_chunks": len(raw_chunks),
                "text": text,
                "text_bm25": bm25_text(text),
                "n_tokens": row["n_tokens"],
                "n_words": len(word_tokens(text)),
                "sentence_start": row["sentence_start"],
                "sentence_end": row["sentence_end"],
                "overlap_tokens": row["overlap_tokens"],
                "overlap_ok": row["overlap_ok"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
            }
        )

    # BM25 view ------------------------------------------------------------
    words = word_tokens(normalised)
    terms = bm25_text(normalised)
    stop_removed = sum(1 for w in words if w.lower() in STOPWORDS)
    stems = sum(
        1
        for w in words
        if w.lower() not in STOPWORDS
        and (
            (T.BN_LETTER_RE.search(w.lower()) and stem_bn(w.lower()) != w.lower())
            or (w.isascii() and w.isalpha() and stem_en(w.lower()) != w.lower())
        )
    )

    section_row = {
        "sec_uid": sec["sec_uid"],
        "act_id": sec["act_id"],
        "act_code": act_code,
        "act_name": sec.get("act_name"),
        "act_name_bn": sec.get("act_name_bn"),
        "section_no": sec.get("section_no"),
        "title": sec.get("title"),
        "chapter": sec.get("chapter"),
        "lang": script_lang,
        "lang_share": round(share, 4),
        "text_norm": normalised,
        "text_bm25": terms,
        "n_chars": len(normalised),
        "n_words": len(words),
        "n_ws_tokens": len(whitespace_tokens(normalised)),
        "n_subword_tokens": tok.count(normalised),
        "n_sentences": len(sentences),
        "n_clauses": clause_total,
        "n_stopwords_removed": stop_removed,
        "n_stems_applied": stems,
        "cleaning": dict(clean_counts),
        "normalization": dict(norm_counts),
    }

    stats = {
        "lang": script_lang,
        "chars_raw": len(raw),
        "chars_norm": len(normalised),
        "words": len(words),
        "ws_tokens": section_row["n_ws_tokens"],
        "subword_tokens": section_row["n_subword_tokens"],
        "sentences": len(sentences),
        "clauses": clause_total,
        "chunks": len(chunks),
        "stop_removed": stop_removed,
        "stems": stems,
        "cleaning": clean_counts,
        "norm": norm_counts,
        "sentence_lengths": [len(s["text"]) for s in sentences],
        "clause_lengths": [len(c) for c in clause_texts],
        "chunk_tokens": [c["n_tokens"] for c in chunks],
        "overlap_chunks": sum(1 for c in chunks if c["overlap_tokens"]),
        "overlap_tokens": sum(c["overlap_tokens"] for c in chunks),
        "overlap_failures": sum(1 for c in chunks if not c["overlap_ok"]),
        "sentence_langs": Counter(
            (s["lang"] if s["lang"] != "neutral" else script_lang) for s in sentences
        ),
        "clause_langs": Counter(
            (s["lang"] if s["lang"] != "neutral" else script_lang)
            for s in segments
            if s["kind"] == "clause"
        ),
        "mixed_sentences": sum(1 for s in sentences if s["lang"] == "mixed"),
        "examples": {},
    }
    return section_row, segments, chunks, stats


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _quantile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return float(ordered[idx])


def _merge_counter(target: Counter, source: Iterable[tuple[str, int]]) -> None:
    for key, value in source:
        target[key] += value


DEMO_QUERIES = (
    "জামিন না পেলে কী হয়?",
    "how long can police keep someone on remand",
    "naraji petition kivabe korbo",
    "এজাহার করতে দেরি হলে কী হবে",
    "cognizable offence e police ki korte pare",
)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "0.0%"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_report(
    stats: dict[str, Any],
    lex_stats: dict[str, Any],
    act_pairs: Sequence[tuple[str, int, int]],
    tokenizer_name: str,
    n_acts: int,
) -> str:
    by_lang = stats["by_lang"]
    n_sections = stats["n_sections"]
    total_sent = stats["n_sentences"]
    total_clause = stats["n_clauses"]
    chunk_tokens = stats["chunk_tokens"]
    sent_lengths = stats["sentence_lengths"]

    lines: list[str] = []
    add = lines.append

    add("# Stage 2 — Text preprocessing report\n")
    add(
        "Generated by `python -m src.stage2`. Input: `data/clean/sections.jsonl` "
        f"({n_sections:,} sections from {n_acts} acts). Subword tokenizer: "
        f"`{tokenizer_name}` (the provisional Stage 3 embedding model; tokenizer "
        "files only, no model weights are loaded).\n"
    )

    # ---- headline ---------------------------------------------------------
    add("## Headline numbers\n")
    add(f"- Sections processed: **{n_sections:,}**")
    add(f"- Sentences: **{stats['n_sentences']:,}**")
    add(f"- Clauses (sub-sections, paragraphs, provisos): **{stats['n_clauses']:,}**")
    add(f"- Retrieval chunks: **{stats['n_chunks']:,}**")
    add(f"- Sections that needed more than one chunk: **{stats['n_chunked_sections']:,}**")
    add(f"- Characters: **{stats['chars_raw']:,}** raw -> **{stats['chars_norm']:,}** normalised")
    add(f"- Whitespace tokens: **{stats['ws_tokens']:,}**")
    add(f"- Subword tokens: **{stats['subword_tokens']:,}**")
    add(f"- Legal lexicon: **{lex_stats['n_terms']}** concepts, **{lex_stats['n_surfaces']}** surface forms")
    add("")

    add("### Pipeline\n")
    add("```")
    add("sections.jsonl")
    add("  -> clean_text()        HTML, page furniture, footnote markers, duplicate section no.")
    add("  -> normalize_text()    NFC, Bangla nukta recomposition, digits -> ASCII, danda -> ।")
    add("  -> detect_lang()       script ratio per section / sentence / clause")
    add("  -> split_all()         sentences (। . ? !) and clauses ((1), (a), provisos)")
    add("  -> tokenise            whitespace baseline + embedding-model subword")
    add("  -> bm25_text()         stopwords + light stemming (BM25 only)")
    add(
        f"  -> chunk_section()     1 section = 1 chunk; >{stats['chunk_max_tokens']} tokens "
        f"-> {stats['chunk_max_tokens']}/{stats['chunk_overlap']} overlap"
    )
    add("```\n")

    # ---- cleaning ---------------------------------------------------------
    clean = stats["clean"]
    add("## 1. Cleaning\n")
    add(
        "**What.** Remove presentation artefacts that are not legal text: HTML "
        "entities/tags, page-number lines, repeated running heads, footnote "
        "markers, superscript footnote digits, `[* * *]` omission markers, and "
        "the section number that duplicates the `section_no` metadata.\n"
    )
    add(
        "**Why.** Footnote numbers and page furniture are corpus-specific noise: "
        "they inflate the vocabulary, create false BM25 matches (a query for "
        "`302` should match section 302, not footnote 302) and waste embedding "
        "tokens. The section number is kept once, in metadata, instead of being "
        "repeated at the head of every chunk.\n"
    )
    add(
        "**What is *not* removed.** Amendment brackets such as "
        "`[জুডিসিয়াল ম্যাজিস্ট্রেট]` are kept: the bracketed words are in force "
        "and are often the exact phrase a query uses. Only numeric-only brackets "
        "at the start of a section, and a footnote digit glued to an opening "
        "bracket (`3[or Police Commissioner]`), are markers and are removed.\n"
    )
    add("**Result.**\n")
    clean_rows = [
        ("HTML entities (`&amp;`, `&nbsp;`)", clean.get("html_entities", 0)),
        ("HTML tags", clean.get("html_tags", 0)),
        ("Zero-width / invisible characters", clean.get("zero_width", 0)),
        ("Line-end hyphenation joins", clean.get("line_hyphenation", 0)),
        ("Page-number lines", clean.get("page_number_lines", 0)),
        ("Page-number dashes (`- 12 -`)", clean.get("page_number_dashes", 0)),
        ("Running-head lines", clean.get("running_headers", 0)),
        ("Duplicate section numbers", clean.get("section_numbers", 0)),
        ("Footnote markers (`[1]`, `3[`, superscripts)", clean.get("footnote_markers", 0)),
        ("Omission markers (`[* * *]`)", clean.get("omitted_passages", 0)),
        ("Sections with runs of >=2 spaces", clean.get("double_spaces", 0)),
    ]
    add(_table(("Rule", "Hits"), clean_rows))
    add("")
    add(
        "HTML, page and running-head counts are **0** in `sections.jsonl` because "
        "Stage 1 already stripped the markup and bdlaws' print HTML has no "
        "pagination furniture. The rules are still applied (and unit-tested on "
        "synthetic input) so that a different source cannot silently reintroduce "
        "them. On this run the rules that fire are: "
        + (", ".join(f"{label} ({count:,})" for label, count in clean_rows if count) or "none")
        + ".\n"
    )
    ex = stats["examples"].get("cleaning")
    if ex:
        add("Example (footnote marker after the section number):\n")
        add("```text")
        add(f"before: {ex['before']}")
        add(f"after:  {ex['after']}")
        add("```\n")
    ex = stats["examples"].get("omitted")
    if ex:
        add("Example (`[* * *]` omission):\n")
        add("```text")
        add(f"before: {ex['before']}")
        add(f"after:  {ex['after']}")
        add("```\n")

    # ---- unicode ----------------------------------------------------------
    norm = stats["norm"]
    add("## 2. Unicode normalisation\n")
    add(
        "**What.** Apply NFC, then explicitly re-compose the three Bangla nukta "
        "letters (ড় ঢ় য়), unify the two danda characters (`৷` -> `।`), and "
        "convert Bengali digits to ASCII (`০১২৩৪৫৬৭৮৯` -> `0123456789`).\n"
    )
    add(
        "**Why NFC alone is not enough for Bangla.** U+09DC (ড়), U+09DD (ঢ়) and "
        "U+09DF (য়) are *composition-excluded*: NFC decomposes them into "
        "base + nukta (ড + ়, ড + ়, য + ়) instead of composing them. The corpus "
        "uses the pre-composed form almost everywhere, and so do Bangla "
        "keyboards. Normalising to the decomposed form would therefore rewrite "
        "thousands of characters and break exact matching against user queries. "
        "The pipeline runs NFC and then maps the decomposed sequences back to "
        "U+09DC/U+09DD/U+09DF, so all three spellings collapse to one code point. "
        "The same function is applied to queries.\n"
    )
    add(
        "**Digits.** Bengali digits are converted in the indexed text only; the "
        "original text stays in `sections.jsonl`. This lets a query for `ধারা ৩০২` "
        "match `section 302`, and lets section references be compared across "
        "languages. Dates and amounts are converted too, which is harmless for "
        "retrieval and keeps the tokenizer vocabulary smaller.\n"
    )
    add("**Result.**\n")
    norm_rows = [
        ("Pre-composed nukta letters in input (ড় ঢ় য়)", norm.get("nukta_precomposed", 0)),
        ("Decomposed nukta sequences in input", norm.get("nukta_decomposed", 0)),
        ("Split vowel signs composed by NFC (ো ৌ)", norm.get("split_vowel_signs", 0)),
        ("Nukta letters re-composed after NFC", norm.get("nukta_recomposed", 0)),
        ("Bengali digits converted", norm.get("bn_digits", 0)),
        ("Danda variants unified (`৷` -> `।`)", norm.get("danda_variants", 0)),
        ("Zero-width characters", norm.get("zero_width", 0)),
    ]
    add(_table(("Operation", "Characters"), norm_rows))
    add("")
    add(
        f"After the pipeline, `normalize_text` is idempotent on "
        f"**{stats['n_canonical']:,}/{n_sections:,}** section texts "
        f"({_pct(stats['n_canonical'], n_sections)}), i.e. no residual variant "
        "forms remain.\n"
    )
    add(
        "Illustration of the variant-character rule (both inputs normalise to the "
        "same output):\n"
    )
    add("```text")
    add("input A: ড়   (U+09DC, pre-composed)")
    add("input B: ড + ় (U+09A1 U+09BC, decomposed)")
    add("output : ড়   (U+09DC)")
    add("```\n")

    # ---- language ---------------------------------------------------------
    add("## 3. Language identification\n")
    add(
        "**What.** Count Bangla letters (U+0980–09FF letter sub-ranges) and Latin "
        "letters in a segment; label `bn` when the Bangla share is >=85%, `en` "
        "when it is <=15%, otherwise `mixed`. Segments with no letters (e.g. "
        "`(1)`, `1973`) inherit the parent's label and are flagged "
        "`lang_inherited`.\n"
    )
    add(
        "**Why.** The corpus is effectively two monolingual sub-corpora, but "
        "legal Bangla routinely embeds English terms (`এজাহার (FIR)`, `চার্জশিট`) "
        "and English sections cite Bangla-free Latin abbreviations. A per-segment "
        "label lets Stage 3 route queries and choose a language-specific "
        "analyzer, and it quantifies code-switching. The 85% threshold is "
        "deliberately conservative: a lower threshold labels an English section "
        "with one Bangla gloss as `mixed`.\n"
    )
    add("**Result.**\n")
    lang_rows = [
        (label, by_lang.get(label, {}).get("sections", 0), stats["lang_sentences"].get(label, 0))
        for label in ("bn", "en", "mixed")
    ]
    add(_table(("Label", "Sections", "Sentences"), lang_rows))
    add("")
    add(
        f"- Code-mixed sentences (both scripts inside one sentence): "
        f"**{stats['n_mixed_sentences']:,}** ({_pct(stats['n_mixed_sentences'], total_sent)} of sentences)."
    )
    add(
        f"- Sections whose text language differs from the Stage 1 `lang` field: "
        f"**{stats['lang_mismatch']:,}** (Stage 1 used `mixed` for 'both "
        "translations present', which is a different concept)."
    )
    add(
        "- Known limitation: romanised Bangla (`naraji`, `jamin`) contains only "
        "Latin letters and is labelled `en`. This is handled at query time by the "
        "colloquial forms in the legal lexicon, not by the script detector."
    )
    add("")
    ex = stats["examples"].get("mixed")
    if ex:
        add(f"Example of a mixed sentence ({ex['sec_uid']}):\n")
        add(f"> {ex['text']}\n")

    # ---- segmentation -----------------------------------------------------
    add("## 4. Sentence and clause segmentation\n")
    add(
        "**What.** Split on the Bangla danda `।` (and `৷`), `?`, `!`, and on "
        "English `.` when followed by whitespace and a capital letter, digit or "
        "Bangla letter. Abbreviations (`No.`, `Sec.`, `s.`, `i.e.`, `U.S.`, "
        "`Cr.P.C.`) and decimals are protected before splitting. Clauses are then "
        "split on sub-section markers `(1)`, `(2)`, `(a)`, `(b)`, `(ক)`, `(খ)` "
        "and on proviso/explanation cues (`Provided that`, `তবে শর্ত থাকে যে`, "
        "`Explanation`, `ব্যাখ্যা`).\n"
    )
    add(
        "**Why.** Sentences are the natural unit for overlap and for "
        "sentence-level citation. Clauses matter more in statutes than in prose: "
        "a proviso can reverse the rule in the main clause, and `(2)` can be the "
        "only thing that distinguishes two otherwise identical provisions. "
        "Keeping the marker attached to the clause text is therefore a hard "
        "requirement; a chunk that starts mid-section must still read "
        "`(2) ...`, not `...`.\n"
    )
    add("**Result.**\n")
    seg_rows = [
        ("Sentences", f"{total_sent:,}"),
        ("Mean sentence length", f"{_mean(sent_lengths):.0f} chars"),
        ("Median sentence length", f"{statistics.median(sent_lengths) if sent_lengths else 0:.0f} chars"),
        ("95th percentile sentence length", f"{_quantile(sent_lengths, 0.95):.0f} chars"),
        ("Longest sentence", f"{max(sent_lengths) if sent_lengths else 0:,} chars"),
        ("Clauses", f"{total_clause:,}"),
        ("Clauses per sentence", f"{total_clause / total_sent:.2f}" if total_sent else "0"),
        ("Clauses with a retained marker", f"{stats['n_marked_clauses']:,}"),
        ("Proviso / explanation clauses", f"{stats['n_proviso_clauses']:,}"),
    ]
    add(_table(("Metric", "Value"), seg_rows))
    add("")
    add("Marker types retained:\n")
    add(_table(("Marker kind", "Count"), sorted(stats["marker_kinds"].items())))
    add("")
    ex = stats["examples"].get("clauses")
    if ex:
        add(f"Example ({ex['sec_uid']}, {ex['lang']}):\n")
        add("```text")
        for c in ex["clauses"][:5]:
            add(f"[{c['marker'] or 'lead'}] ({c['clause_type']}) {c['text'][:140]}")
        add("```\n")
    ex = stats["examples"].get("sentences")
    if ex:
        add(f"English sentence split example ({ex['sec_uid']}):\n")
        add("```text")
        for s in ex["sentences"][:3]:
            add(f"- {s[:160]}")
        add("```\n")

    # ---- tokenisation -----------------------------------------------------
    add("## 5. Tokenisation: whitespace vs subword\n")
    add(
        "**What.** Compare `text.split()` (whitespace) with the embedding "
        "model's subword tokenizer. Counts are computed per section and "
        "aggregated by language.\n"
    )
    add(
        "**Why.** Chunk size and cost are measured in *subword* tokens, not "
        "words. Bangla is agglutinative and the multilingual tokenizer has far "
        "less Bangla in its vocabulary than English, so the same legal content "
        "can cost two to three times as many tokens. The comparison sets the "
        "chunk budget and predicts the cost of the Bangla half of the corpus.\n"
    )
    add(
        "**A Bangla-specific pitfall.** `re.findall(r\"\\w+\", \"বাংলা\")` returns "
        "four single-letter tokens because Bangla combining marks are not "
        "`\\w`. The pipeline therefore splits on whitespace and trims edge "
        "punctuation, which keeps conjuncts intact.\n"
    )
    add("**Result.**\n")
    tok_rows = []
    for label in ("en", "bn"):
        d = by_lang.get(label, {})
        ws = d.get("ws", 0)
        sub = d.get("sub", 0)
        words = d.get("words", 0)
        chars = d.get("chars", 0)
        tok_rows.append(
            (
                label,
                f"{d.get('sections', 0):,}",
                f"{words:,}",
                f"{ws:,}",
                f"{sub:,}",
                f"{sub / words:.2f}" if words else "0",
                f"{100 * sub / chars:.1f}" if chars else "0",
            )
        )
    tok_rows.append(
        (
            "all",
            f"{n_sections:,}",
            f"{stats['words']:,}",
            f"{stats['ws_tokens']:,}",
            f"{stats['subword_tokens']:,}",
            f"{stats['subword_tokens'] / stats['words']:.2f}" if stats["words"] else "0",
            f"{100 * stats['subword_tokens'] / stats['chars_norm']:.1f}" if stats["chars_norm"] else "0",
        )
    )
    add(_table(("Lang", "Sections", "Words", "Whitespace", "Subword", "Sub/word", "Sub/100 chars"), tok_rows))
    add("")
    add(
        f"Overall, the subword tokenizer produces **{stats['subword_tokens'] / max(1, stats['ws_tokens']):.2f}x** "
        f"more tokens than whitespace splitting. For Bangla the ratio is "
        f"**{by_lang.get('bn', {}).get('sub', 0) / max(1, by_lang.get('bn', {}).get('ws', 1)):.2f}x**"
        + (
            f", versus **{by_lang['en']['sub'] / max(1, by_lang['en']['ws']):.2f}x** for English.\n"
            if by_lang.get("en", {}).get("ws")
            else "; this run contains no English sections, so the English ratio is not estimated.\n"
        )
    )
    add(f"Parallel check on the {len(act_pairs)} act titles (the only official EN/BN pairs in the corpus):\n")
    if act_pairs:
        en_total = sum(p[1] for p in act_pairs)
        bn_total = sum(p[2] for p in act_pairs)
        add(
            f"- English titles: **{en_total:,}** tokens; Bangla titles: **{bn_total:,}** tokens; "
            f"ratio **{bn_total / en_total:.2f}x**."
        )
        add("")
        add(_table(("Act", "EN tokens", "BN tokens"), [(p[0][:55], p[1], p[2]) for p in act_pairs[:8]]))
        add("")
    add("Illustrative token costs:\n")
    add(_table(("Text", "Whitespace", "Subword"), stats["examples"]["token_terms"]))
    add("")
    max_tok = stats["chunk_max_tokens"]
    bn_ratio = by_lang.get("bn", {}).get("sub", 0) / max(1, by_lang.get("bn", {}).get("words", 1))
    en_ratio = by_lang.get("en", {}).get("sub", 0) / max(1, by_lang.get("en", {}).get("words", 1))
    bn_words = int(max_tok / bn_ratio) if bn_ratio else 0
    en_words = int(max_tok / en_ratio) if en_ratio else 0
    add(
        f"**Implication.** A {max_tok}-subword-token chunk holds roughly "
        f"{bn_words} Bangla words but {en_words} English words at the ratios "
        "measured above. The budget is therefore conservative for English and "
        "appropriate for Bangla; it also stays inside a 512-token model limit "
        "after the query and special tokens.\n"
    )

    # ---- bm25 -------------------------------------------------------------
    bm = stats["bm25"]
    add("## 6. Stopwords and stemming (BM25 only)\n")
    add(
        f"**What.** For the BM25 view only: remove a {len(BN_STOPWORDS)}-word Bangla stopword "
        "list and a small English function-word list, then apply a light "
        "suffix stemmer (Bangla) and plural-only normalisation (English). Dense "
        "embeddings receive the normalised text with stopwords intact.\n"
    )
    add(
        "**Why.** BM25 is a lexical matcher: frequent function words such as "
        "`এই`, `এবং`, `যে`, `of`, `the` add no discriminative power and lengthen "
        "every document, which dilutes the score. Inflection is a real problem in "
        "Bangla (`আইনের`, `আইনকে`, `আইনে` all refer to `আইন`), so a light stemmer "
        "improves recall. Dense models already handle both phenomena in context, "
        "and removing stopwords there can delete legally meaningful negation "
        "(`না`, `নহে`, `not`), so the two views are deliberately different.\n"
    )
    add("**Result.**\n")
    add(
        _table(
            ("Metric", "Value"),
            [
                ("Bangla stopwords", len(BN_STOPWORDS)),
                ("English stopwords", len(EN_STOPWORDS)),
                ("Tokens before BM25 filtering", f"{bm['words']:,}"),
                ("Tokens after stopword removal", f"{bm['words'] - bm['stop_removed']:,}"),
                ("Stopwords removed", f"{bm['stop_removed']:,} ({_pct(bm['stop_removed'], bm['words'])})"),
                ("Tokens changed by stemming", f"{bm['stems']:,}"),
                ("Vocabulary before", f"{bm['vocab_before']:,}"),
                ("Vocabulary after", f"{bm['vocab_after']:,}"),
            ],
        )
    )
    add("")
    add("Stemmer examples:\n")
    add(_table(("Surface", "Stem", "Result"), stats["examples"]["stems"]))
    add("")
    add(
        "The Bangla stemmer is intentionally shallow: single-character suffixes "
        "such as `য়` and `র` are **not** stripped, because doing so turns "
        "`বিষয়` into `বিষ` and `ঘর` into `ঘ`. Longest-suffix-first ordering and a "
        "three-character minimum keep it safe. English uses plural-only "
        "normalisation; Porter-style stemming would conflate `conviction` and "
        "`convict`, which are legally distinct.\n"
    )

    # ---- chunking ---------------------------------------------------------
    add("## 7. Chunking\n")
    add(
        f"**What.** One section is one chunk. A section longer than "
        f"{stats['chunk_max_tokens']} subword tokens (including its citation "
        f"header) is split on sentence boundaries into chunks of at most "
        f"{stats['chunk_max_tokens']} tokens with about {stats['chunk_overlap']} "
        "tokens of overlap. Every chunk keeps `sec_uid` and gets "
        "`chunk_id = {sec_uid}#cNN`. A sentence that alone exceeds the budget is "
        "split on word boundaries; a clause is never cut without its marker.\n"
    )
    add(
        "**Why.** Section-level chunks preserve the legal unit of meaning and "
        f"make citation exact. The {stats['chunk_max_tokens']}/{stats['chunk_overlap']} "
        "budget keeps chunks inside the embedding model's context window, and "
        "overlap protects a provision that straddles a boundary. Splitting on "
        "sentences rather than a fixed character window avoids chunks that begin "
        "mid-clause.\n"
    )
    add("**Result.**\n")
    add(
        _table(
            ("Metric", "Value"),
            [
                ("Sections", f"{n_sections:,}"),
                ("Sections kept as one chunk", f"{n_sections - stats['n_chunked_sections']:,}"),
                ("Sections split into >1 chunk", f"{stats['n_chunked_sections']:,}"),
                ("Chunks", f"{stats['n_chunks']:,}"),
                (
                    "Mean chunks per split section",
                    f"{(stats['n_chunks'] - (n_sections - stats['n_chunked_sections'])) / max(1, stats['n_chunked_sections']):.2f}",
                ),
                ("Mean chunk size", f"{_mean(chunk_tokens):.0f} tokens"),
                ("Median chunk size", f"{statistics.median(chunk_tokens) if chunk_tokens else 0:.0f} tokens"),
                ("Largest chunk", f"{max(chunk_tokens) if chunk_tokens else 0} tokens"),
                (
                    f"Chunks over {stats['chunk_max_tokens']} tokens",
                    f"{sum(1 for x in chunk_tokens if x > stats['chunk_max_tokens']):,}",
                ),
                ("Chunks with overlap", f"{stats['overlap_chunks']:,}"),
                ("Mean overlap", f"{stats['overlap_tokens'] / max(1, stats['overlap_chunks']):.0f} tokens"),
                ("Overlap verification failures", f"{stats['overlap_failures']:,}"),
            ],
        )
    )
    add("")
    add(
        f"Every chunk carries a `sec_uid`; **{stats['chunk_uid_coverage']:,}/{stats['n_chunks']:,}** "
        "chunks pass that check, and the largest chunk is within the budget. "
        "Chunk ids are unique.\n"
    )
    ex = stats["examples"].get("chunks")
    if ex:
        add(f"Example: `{ex['sec_uid']}` ({ex['n_chunks']} chunks, {ex['lang']}).\n")
        add("```text")
        for c in ex["chunks"][:3]:
            add(f"[{c['chunk_id']}] {c['n_tokens']} tokens, overlap={c['overlap_tokens']}")
            add(f"  {c['text'][:220].replace(chr(10), ' | ')}")
        add("```\n")

    # ---- lexicon ----------------------------------------------------------
    add("## 8. Legal dictionary and query expansion\n")
    add(
        "**What.** A hand-built lexicon of "
        f"**{lex_stats['n_terms']}** concepts with formal English, formal Bangla "
        f"and colloquial/romanised forms ({lex_stats['n_surfaces']} surface forms "
        f"in total), cross-links between concepts, and "
        f"{lex_stats['n_statutes']} statute references (validated against "
        "`sec_uid`). Query expansion matches the longest surface form, ignores "
        "overlaps, and adds cross-lingual synonyms plus a few related concepts.\n"
    )
    add(
        "**Why.** Users do not speak in statute language. They write `jamin`, "
        "`naraji`, `rimand`, `amaljoggo` or `এজাহার`, while the corpus says "
        "`জামিন`, `নারাজি`, `পুনঃসমর্পণ`, `আমলযোগ্য` or `FIR`. Expansion bridges "
        "the three registers and the two scripts, and the statute links give "
        "Stage 3 an exact-section fallback when lexical matching fails. Expansion "
        "is kept small and is meant to be weighted below the original query "
        "terms, so it improves recall without hijacking the query.\n"
    )
    add("**Result.**\n")
    add(
        _table(
            ("Property", "Value"),
            [
                ("Concepts", lex_stats["n_terms"]),
                ("English forms", lex_stats["n_en_forms"]),
                ("Bangla forms", lex_stats["n_bn_forms"]),
                ("Colloquial forms", lex_stats["n_colloquial"]),
                ("Concept links", lex_stats["n_related"]),
                ("Statute references", lex_stats["n_statutes"]),
                ("Unknown statute references", len(lex_stats["unknown_refs"])),
            ],
        )
    )
    add("")
    add("Expansion examples:\n")
    for demo in lex_stats["demos"]:
        concepts = ", ".join(f"`{c['id']}`" for c in demo["concepts"]) or "—"
        extra = ", ".join(demo["expansion_terms"][:8]) or "—"
        add(f"- Query: **{demo['query']}**")
        add(f"  - concepts: {concepts}")
        add(f"  - expansion: {extra}")
        add(f"  - statute refs: {', '.join(demo['statute_refs']) or '—'}")
    add("")

    # ---- outputs ----------------------------------------------------------
    add("## 9. Outputs\n")
    add(
        _table(
            ("File", "Rows", "Purpose"),
            [
                ("`data/clean/sections_norm.jsonl`", f"{n_sections:,}", "Canonical text + BM25 text + counts per section"),
                ("`data/clean/segments.jsonl`", f"{stats['n_segments']:,}", "Sentence and clause spans with stable ids"),
                ("`data/clean/chunks.jsonl`", f"{stats['n_chunks']:,}", "Retrieval units, `sec_uid` preserved"),
                ("`results/stage2_token_stats.csv`", f"{n_sections:,}", "Per-section token counts"),
                ("`results/stage2_summary.json`", "1", "Aggregate numbers used in this report"),
            ],
        )
    )
    add("")
    add("`sections_norm.jsonl` fields:\n")
    add(
        _table(
            ("Field", "Meaning"),
            [
                ("`sec_uid`", "Stable section id (join key to `sections.jsonl`)"),
                ("`lang`", "`bn`, `en` or `mixed` from the script check"),
                ("`lang_share`", "Bangla share of letters"),
                ("`text_norm`", "Cleaned, normalised text (dense input)"),
                ("`text_bm25`", "Stopword-removed, stemmed terms (BM25 input only)"),
                ("`n_chars`, `n_words`, `n_ws_tokens`, `n_subword_tokens`", "Size metrics"),
                ("`n_sentences`, `n_clauses`", "Segmentation counts"),
                ("`n_stopwords_removed`, `n_stems_applied`", "BM25-only effects"),
                ("`cleaning`, `normalization`", "Per-rule change counts for audit"),
            ],
        )
    )
    add("")
    add("`chunks.jsonl` fields:\n")
    add(
        _table(
            ("Field", "Meaning"),
            [
                ("`chunk_id`", "`{sec_uid}#cNN`, unique"),
                ("`sec_uid`", "Section id, present on every chunk"),
                ("`text`", "Citation header + body"),
                ("`text_bm25`", "BM25 view of `text`"),
                ("`n_tokens`, `n_words`", "Subword and word counts"),
                ("`sentence_start`, `sentence_end`", "Inclusive sentence span of the new content"),
                ("`overlap_tokens`, `overlap_ok`", "Tokens of previous-chunk context prepended, and a verification flag"),
                ("`char_start`, `char_end`", "Body span in `text_norm`"),
            ],
        )
    )
    add("")

    # ---- validation -------------------------------------------------------
    add("## 10. Validation and limitations\n")
    add(
        "Checks run at the end of the pipeline:\n"
        "- Every section with text produced at least one chunk.\n"
        "- Every chunk has a non-empty `sec_uid`; chunk ids are unique.\n"
        "- No chunk exceeds the configured subword-token budget.\n"
        "- Every overlap region is a suffix of the previous chunk body.\n"
        "- `normalize_text` is idempotent on every section.\n"
        "- All statute references in the lexicon resolve to a real `sec_uid`.\n"
        "- Unit tests cover footnote stripping, nukta equivalence, digit "
        "unification, language ID, sentence/clause splitting, chunk budgets and "
        "query expansion (`python -m pytest tests/test_stage2.py`).\n"
    )
    add("Known limitations:\n")
    add(
        "- **Script detection is not language detection.** Romanised Bangla is "
        "labelled `en`; the lexicon compensates at query time.\n"
        f"- **No parallel translations.** Only {len(act_pairs)} act titles are officially "
        "bilingual, so the EN/BN token comparison is exact for titles and "
        "aggregate (not sentence-aligned) for the rest.\n"
        "- **The tokenizer is provisional.** It is the Stage 3 embedding "
        "model's tokenizer; if Stage 3 chooses a different model, chunk sizes "
        "must be recomputed with `--tokenizer-model`.\n"
        "- **Page/header rules are defensive.** They fire on synthetic input "
        "and on other sources, not on the current bdlaws extract.\n"
        "- **Light stemming is not morphological analysis.** It is tuned for "
        "BM25 recall, not for linguistic correctness.\n"
    )

    add("## 11. Reproduction\n")
    add("```bash")
    add("python -m src.stage2                      # full corpus")
    add("python -m src.stage2 --tokenizer-model BAAI/bge-m3")
    add("python -m src.stage2 --limit 50           # smoke test -> data/clean/smoke")
    add("python -m pytest tests/test_stage2.py -q")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="process only the first N sections (smoke test)")
    parser.add_argument("--tokenizer-model", default=C.EMBED_MODEL, help="HF tokenizer name, or 'none'")
    parser.add_argument("--max-tokens", type=int, default=C.CHUNK_MAX_TOKENS)
    parser.add_argument("--overlap", type=int, default=C.CHUNK_OVERLAP_TOKENS)
    args = parser.parse_args(argv)

    sections_path = C.DATA_CLEAN / "sections.jsonl"
    if not sections_path.exists():
        print(f"missing {sections_path}; run `python -m src.preprocess` first")
        return 1

    all_sections = read_jsonl(sections_path)
    sections = all_sections[: args.limit] if args.limit else all_sections
    acts = read_jsonl(C.DATA_CLEAN / "acts.jsonl")
    act_code = {a["act_id"]: a["code"] for a in acts}

    lexicon_path = C.DATA_LEXICON / "legal_terms.json"
    terms: list[LegalTerm] = load_lexicon(lexicon_path) if lexicon_path.exists() else []
    if not terms:
        print(f"[warn] no lexicon at {lexicon_path}")

    tok = load_tokenizer(args.tokenizer_model)
    print(f"[info] tokenizer: {tok.name}")
    print(f"[info] sections: {len(sections):,}; lexicon: {len(terms)} concepts")

    out_dir = C.DATA_CLEAN if not args.limit else C.DATA_CLEAN / "smoke"
    report_path = C.REPORT / ("stage2_report.md" if not args.limit else "stage2_smoke.md")
    out_dir.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)

    section_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []

    agg: dict[str, Any] = {
        "clean": Counter(),
        "norm": Counter(),
        "lang_sections": Counter(),
        "lang_sentences": Counter(),
        "lang_clauses": Counter(),
        "marker_kinds": Counter(),
        "by_lang": defaultdict(lambda: {"sections": 0, "words": 0, "ws": 0, "sub": 0, "chars": 0}),
    }
    sentence_lengths: list[int] = []
    clause_lengths: list[int] = []
    chunk_tokens: list[int] = []
    words_total = 0
    ws_total = 0
    sub_total = 0
    chars_raw = 0
    chars_norm = 0
    stop_total = 0
    stem_total = 0
    n_marked = 0
    n_proviso = 0
    n_mixed_sentences = 0
    n_chunked_sections = 0
    overlap_chunks = 0
    overlap_tokens = 0
    overlap_failures = 0
    vocab_before: set[str] = set()
    vocab_after: set[str] = set()
    examples: dict[str, Any] = {}
    lang_mismatch = 0

    for k, sec in enumerate(sections, start=1):
        row, segs, chunks, st = process_section(
            sec, tok, act_code.get(sec["act_id"], ""), args.max_tokens, args.overlap
        )
        section_rows.append(row)
        segment_rows.extend(segs)
        chunk_rows.extend(chunks)

        _merge_counter(agg["clean"], st["cleaning"].items())
        _merge_counter(agg["norm"], st["norm"].items())
        agg["lang_sections"][st["lang"]] += 1
        agg["lang_sentences"].update(st["sentence_langs"])
        agg["lang_clauses"].update(st["clause_langs"])
        for seg in segs:
            if seg["kind"] != "clause":
                continue
            if seg["clause_type"] in {"proviso", "explanation"}:
                n_proviso += 1
            if seg["marker"]:
                n_marked += 1
                marker = str(seg["marker"])
                kind = "digit" if marker[0].isdigit() or "\u09E6" <= marker[0] <= "\u09EF" else "letter"
                agg["marker_kinds"][f"({kind})"] += 1
        if sec.get("lang") and sec.get("lang") != st["lang"]:
            lang_mismatch += 1

        d = agg["by_lang"][st["lang"]]
        d["sections"] += 1
        d["words"] += st["words"]
        d["ws"] += st["ws_tokens"]
        d["sub"] += st["subword_tokens"]
        d["chars"] += st["chars_norm"]

        sentence_lengths.extend(st["sentence_lengths"])
        clause_lengths.extend(st["clause_lengths"])
        chunk_tokens.extend(st["chunk_tokens"])
        words_total += st["words"]
        ws_total += st["ws_tokens"]
        sub_total += st["subword_tokens"]
        chars_raw += st["chars_raw"]
        chars_norm += st["chars_norm"]
        stop_total += st["stop_removed"]
        stem_total += st["stems"]
        n_mixed_sentences += st["mixed_sentences"]
        if st["chunks"] > 1:
            n_chunked_sections += 1
        overlap_chunks += st["overlap_chunks"]
        overlap_tokens += st["overlap_tokens"]
        overlap_failures += st["overlap_failures"]

        for w in word_tokens(row["text_norm"]):
            vocab_before.add(w.lower())
        vocab_after.update(row["text_bm25"].split())

        if "cleaning" not in examples and st["cleaning"].get("footnote_markers"):
            raw = sec.get("text_bn") or sec.get("text_en") or ""
            m = T.GLUED_FN_RE.search(raw)
            if m:
                before = raw[max(0, m.start() - 80) : m.end() + 80].replace("\n", " ")
                after = T.GLUED_FN_RE.sub("[", before)
            else:
                m = re.search(r"\[\s*(?:\d{1,3}|[০-৯]{1,3})\s*\]", raw)
                if m:
                    before = raw[max(0, m.start() - 80) : m.end() + 80].replace("\n", " ")
                    after = (raw[: m.start()] + raw[m.end() :])[
                        max(0, m.start() - 80) : m.start() + 80
                    ]
                else:
                    before = raw[:180].replace("\n", " ")
                    after = row["text_norm"][:180]
            examples["cleaning"] = {
                "sec_uid": sec["sec_uid"],
                "before": before.strip(),
                "after": after.strip(),
            }
        if "omitted" not in examples and st["cleaning"].get("omitted_passages"):
            raw = sec.get("text_bn") or sec.get("text_en") or ""
            m = T.OMITTED_RE.search(raw)
            if m:
                before = raw[max(0, m.start() - 90) : m.end() + 90].replace("\n", " ")
                tail = raw[m.end() : m.end() + 40].strip()
                idx = row["text_norm"].find(tail[:15]) if tail else -1
                after = (
                    row["text_norm"][max(0, idx - 90) : idx + 90]
                    if idx >= 0
                    else row["text_norm"][:180]
                )
                examples["omitted"] = {
                    "sec_uid": sec["sec_uid"],
                    "before": before.strip(),
                    "after": after.strip(),
                }
        if "mixed" not in examples:
            for s in segs:
                if s["kind"] == "sentence" and s["lang"] == "mixed":
                    examples["mixed"] = {"sec_uid": sec["sec_uid"], "text": s["text"][:220]}
                    break
        if "clauses" not in examples and st["lang"] == "bn" and st["clauses"] >= 3:
            clauses = [s for s in segs if s["kind"] == "clause"][:6]
            examples["clauses"] = {"sec_uid": sec["sec_uid"], "lang": st["lang"], "clauses": clauses}
        if "sentences" not in examples and sec["sec_uid"] == "pc1860_s302":
            examples["sentences"] = {
                "sec_uid": sec["sec_uid"],
                "sentences": [s["text"] for s in segs if s["kind"] == "sentence"],
            }
        if "chunks" not in examples and st["chunks"] > 1 and len(chunks[0]["text"]) > 300:
            examples["chunks"] = {
                "sec_uid": sec["sec_uid"],
                "lang": st["lang"],
                "n_chunks": st["chunks"],
                "chunks": chunks,
            }

        if k % 250 == 0:
            print(f"  {k:5d}/{len(sections):,} sections, {len(chunk_rows):,} chunks")

    # ---- final statistics -------------------------------------------------
    n_chunks = len(chunk_rows)
    chunk_uid_ok = sum(1 for c in chunk_rows if c.get("sec_uid"))
    chunk_ids = {c["chunk_id"] for c in chunk_rows}
    chunk_uid_coverage = chunk_uid_ok
    n_canonical = sum(1 for r in section_rows if T.is_canonical(r["text_norm"]))

    # token examples for the report
    example_terms = [
        "এজাহার",
        "জামিন",
        "পুনঃসমর্পণ",
        "আমলযোগ্য",
        "FIR",
        "bail",
        "remand",
        "cognizable offence",
    ]
    token_terms = [
        (text, len(text.split()), tok.count(text)) for text in example_terms
    ]

    stem_examples: list[tuple[str, str, str]] = []
    for surface in ("আইনের", "আইনকে", "অভিযুক্তকে", "সাক্ষীদের", "মামলার", "বিষয়"):
        stem = stem_bn(surface)
        stem_examples.append(
            (surface, stem, "stemmed" if stem != surface else "unchanged (guard)")
        )
    for surface in ("offences", "sections", "bails", "accused"):
        stem = stem_en(surface)
        stem_examples.append(
            (surface, stem, "stemmed" if stem != surface else "unchanged (not a plural)")
        )

    lex_stats: dict[str, Any] = {
        "n_terms": len(terms),
        "n_en_forms": sum(len(t.en) for t in terms),
        "n_bn_forms": sum(len(t.bn) for t in terms),
        "n_colloquial": sum(len(t.colloquial) for t in terms),
        "n_related": sum(len(t.related) for t in terms),
        "n_statutes": len({uid for t in terms for uid in t.statutes}),
        "unknown_refs": validate_statutes(terms, (s["sec_uid"] for s in all_sections)),
        "demos": [expand_query(q, terms) for q in DEMO_QUERIES],
    }
    lex_stats["n_surfaces"] = (
        lex_stats["n_en_forms"] + lex_stats["n_bn_forms"] + lex_stats["n_colloquial"]
    )

    # parallel act titles
    act_pairs: list[tuple[str, int, int]] = []
    for meta in acts:
        if meta.get("repealed"):
            continue
        if meta.get("act_name") and meta.get("act_name_bn"):
            act_pairs.append(
                (meta["act_name"], tok.count(meta["act_name"]), tok.count(meta["act_name_bn"]))
            )

    stats: dict[str, Any] = {
        "n_sections": len(section_rows),
        "n_chunks": n_chunks,
        "chunk_max_tokens": args.max_tokens,
        "chunk_overlap": args.overlap,
        "n_chunked_sections": n_chunked_sections,
        "n_segments": len(segment_rows),
        "n_sentences": sum(r["n_sentences"] for r in section_rows),
        "n_clauses": sum(r["n_clauses"] for r in section_rows),
        "n_marked_clauses": n_marked,
        "n_proviso_clauses": n_proviso,
        "n_mixed_sentences": n_mixed_sentences,
        "n_canonical": n_canonical,
        "lang_mismatch": lang_mismatch,
        "chars_raw": chars_raw,
        "chars_norm": chars_norm,
        "words": words_total,
        "ws_tokens": ws_total,
        "subword_tokens": sub_total,
        "clean": dict(agg["clean"]),
        "norm": dict(agg["norm"]),
        "lang_sections": dict(agg["lang_sections"]),
        "lang_sentences": dict(agg["lang_sentences"]),
        "lang_clauses": dict(agg["lang_clauses"]),
        "marker_kinds": dict(agg["marker_kinds"]),
        "by_lang": {k: dict(v) for k, v in agg["by_lang"].items()},
        "sentence_lengths": sentence_lengths,
        "clause_lengths": clause_lengths,
        "chunk_tokens": chunk_tokens,
        "overlap_chunks": overlap_chunks,
        "overlap_tokens": overlap_tokens,
        "overlap_failures": overlap_failures,
        "chunk_uid_coverage": chunk_uid_coverage,
        "bm25": {
            "words": words_total,
            "stop_removed": stop_total,
            "stems": stem_total,
            "vocab_before": len(vocab_before),
            "vocab_after": len(vocab_after),
        },
        "examples": {
            **examples,
            "token_terms": token_terms,
            "stems": stem_examples,
        },
    }

    # ---- integrity checks -------------------------------------------------
    problems: list[str] = []
    if n_chunks == 0:
        problems.append("no chunks produced")
    if len(chunk_ids) != n_chunks:
        problems.append("duplicate chunk_id values")
    if chunk_uid_coverage != n_chunks:
        problems.append("chunk missing sec_uid")
    if any(c["n_tokens"] > args.max_tokens for c in chunk_rows):
        problems.append("chunk over token budget")
    if overlap_failures:
        problems.append(f"{overlap_failures} overlap verification failures")
    if n_canonical != len(section_rows):
        problems.append("normalize_text is not idempotent on some sections")
    if problems:
        raise SystemExit("stage2 integrity failure: " + "; ".join(problems))

    # ---- write outputs ----------------------------------------------------
    n_sec = write_jsonl(out_dir / "sections_norm.jsonl", section_rows)
    n_seg = write_jsonl(out_dir / "segments.jsonl", segment_rows)
    n_chunk = write_jsonl(out_dir / "chunks.jsonl", chunk_rows)

    chunks_per_section = Counter(c["sec_uid"] for c in chunk_rows)
    with (C.RESULTS / ("stage2_token_stats.csv" if not args.limit else "stage2_smoke_token_stats.csv")).open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sec_uid",
                "lang",
                "chars",
                "words",
                "ws_tokens",
                "subword_tokens",
                "sentences",
                "clauses",
                "chunks",
                "stopwords_removed",
                "stems_applied",
            ]
        )
        for r in section_rows:
            writer.writerow(
                [
                    r["sec_uid"],
                    r["lang"],
                    r["n_chars"],
                    r["n_words"],
                    r["n_ws_tokens"],
                    r["n_subword_tokens"],
                    r["n_sentences"],
                    r["n_clauses"],
                    chunks_per_section.get(r["sec_uid"], 0),
                    r["n_stopwords_removed"],
                    r["n_stems_applied"],
                ]
            )

    summary = {
        "tokenizer": tok.name,
        "chunk_max_tokens": args.max_tokens,
        "chunk_overlap_tokens": args.overlap,
        "sections": len(section_rows),
        "segments": len(segment_rows),
        "chunks": n_chunks,
        "counts": {
            "chars_raw": chars_raw,
            "chars_norm": chars_norm,
            "words": words_total,
            "ws_tokens": ws_total,
            "subword_tokens": sub_total,
            "sentences": stats["n_sentences"],
            "clauses": stats["n_clauses"],
        },
        "by_lang": stats["by_lang"],
        "cleaning": stats["clean"],
        "normalization": stats["norm"],
        "bm25": stats["bm25"],
        "lexicon": {k: v for k, v in lex_stats.items() if k != "demos"},
    }
    (C.RESULTS / ("stage2_summary.json" if not args.limit else "stage2_smoke_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = build_report(stats, lex_stats, act_pairs, tok.name, len({s["act_id"] for s in sections}))
    report_path.write_text(report, encoding="utf-8")

    print(f"\n[done] sections={n_sec:,} segments={n_seg:,} chunks={n_chunk:,}")
    print(f"[done] {out_dir / 'chunks.jsonl'}")
    print(f"[done] {report_path}")
    print(
        f"[stats] chars {chars_raw:,}->{chars_norm:,} | words {words_total:,} | "
        f"ws {ws_total:,} | subword {sub_total:,} | chunks {n_chunks:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
