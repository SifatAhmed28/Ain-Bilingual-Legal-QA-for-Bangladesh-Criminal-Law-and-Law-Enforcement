"""Legal term lexicon and query expansion.

The lexicon is a small, hand-built bridge between the three ways users refer to
the same concept:

* formal English (``remand``, ``cognizable offence``),
* formal Bangla (``পুনঃসমর্পণ``, ``আমলযোগ্য অপরাধ``),
* colloquial / romanised Bangla (``rimand``, ``amaljoggo``, ``jamin``).

Expansion is deliberately conservative: a concept contributes at most a few
cross-lingual synonyms, and expansion terms are meant to be added to a BM25
query with a lower weight (or used as a second dense query), never to replace
the user's words.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .textnorm import normalize_text


@dataclass(frozen=True)
class LegalTerm:
    id: str
    label_en: str
    label_bn: str
    en: tuple[str, ...]
    bn: tuple[str, ...]
    colloquial: tuple[str, ...]
    related: tuple[str, ...]
    statutes: tuple[str, ...]

    @property
    def surfaces(self) -> tuple[str, ...]:
        return (*self.en, *self.bn, *self.colloquial)


def load_lexicon(path: Path) -> list[LegalTerm]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    terms: list[LegalTerm] = []
    seen: set[str] = set()
    for row in raw["terms"]:
        for key in ("id", "label_en", "label_bn", "en", "bn", "colloquial", "related", "statutes"):
            if key not in row:
                raise ValueError(f"lexicon entry missing {key!r}: {row.get('id')}")
        if row["id"] in seen:
            raise ValueError(f"duplicate lexicon id: {row['id']}")
        if not row["en"] or not row["bn"]:
            raise ValueError(f"lexicon entry {row['id']} needs at least one en and one bn form")
        seen.add(row["id"])
        terms.append(
            LegalTerm(
                id=row["id"],
                label_en=row["label_en"],
                label_bn=row["label_bn"],
                en=tuple(row["en"]),
                bn=tuple(row["bn"]),
                colloquial=tuple(row["colloquial"]),
                related=tuple(row["related"]),
                statutes=tuple(row["statutes"]),
            )
        )
    ids = {t.id for t in terms}
    missing = [(t.id, r) for t in terms for r in t.related if r not in ids]
    if missing:
        raise ValueError(f"lexicon related links point at unknown ids: {missing}")
    return terms


def _norm(text: str) -> str:
    return normalize_text(text)[0].casefold()


def _latin_pattern(form: str) -> re.Pattern[str]:
    body = re.escape(form).replace(r"\ ", r"\s+")
    # Allow a regular English inflection on the final word (bail -> bails,
    # arrested -> arrest is covered by listing the inflected form explicitly).
    return re.compile(rf"(?<![A-Za-z]){body}(?:s|es|ed|ing)?(?![A-Za-z])", re.I)


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    term: LegalTerm
    surface: str


def find_matches(query: str, terms: Sequence[LegalTerm]) -> list[Match]:
    """Find lexicon concepts in a query, longest surface form wins."""
    q = _norm(query)
    hits: list[Match] = []
    for term in terms:
        for form in (*term.en, *term.colloquial):
            for m in _latin_pattern(form).finditer(q):
                hits.append(Match(m.start(), m.end(), term, form))
        for form in term.bn:
            needle = _norm(form)
            start = q.find(needle)
            while start != -1:
                hits.append(Match(start, start + len(needle), term, form))
                start = q.find(needle, start + 1)

    hits.sort(key=lambda h: (-(h.end - h.start), h.start))
    chosen: list[Match] = []
    occupied: list[tuple[int, int]] = []
    for hit in hits:
        if any(not (hit.end <= a or hit.start >= b) for a, b in occupied):
            continue
        occupied.append((hit.start, hit.end))
        chosen.append(hit)
    chosen.sort(key=lambda h: h.start)
    return chosen


def expand_query(
    query: str,
    terms: Sequence[LegalTerm],
    max_per_concept: int = 4,
    include_related: bool = True,
) -> dict[str, object]:
    """Return matched concepts plus cross-lingual expansion terms.

    The result is JSON-serialisable so it can be logged next to retrieval
    results for error analysis.
    """
    q = _norm(query)
    by_id = {t.id: t for t in terms}
    matches = find_matches(query, terms)

    concepts: list[dict[str, object]] = []
    expansion: list[str] = []
    seen_concepts: set[str] = set()
    seen_terms: set[str] = set()

    def add_terms(term: LegalTerm, limit: int) -> list[str]:
        added: list[str] = []
        for form in term.surfaces:
            nf = _norm(form)
            if not nf or nf in q or nf in seen_terms:
                continue
            seen_terms.add(nf)
            added.append(form)
            expansion.append(form)
            if len(added) >= limit:
                break
        return added

    for hit in matches:
        term = hit.term
        if term.id in seen_concepts:
            continue
        seen_concepts.add(term.id)
        extra = add_terms(term, max_per_concept)
        related_ids = [rid for rid in term.related if rid in by_id]
        if include_related:
            for rid in related_ids:
                add_terms(by_id[rid], 2)
        concepts.append(
            {
                "id": term.id,
                "label_en": term.label_en,
                "label_bn": term.label_bn,
                "matched": hit.surface,
                "expansion": extra,
                "related": related_ids,
                "statutes": list(term.statutes),
            }
        )

    statutes: list[str] = []
    for c in concepts:
        for uid in c["statutes"]:  # type: ignore[index]
            if uid not in statutes:
                statutes.append(uid)

    return {
        "query": query,
        "concepts": concepts,
        "expansion_terms": expansion,
        "statute_refs": statutes,
    }


def expanded_query_text(query: str, expansion: dict[str, object]) -> str:
    """Append expansion terms to the original query for a BM25 run."""
    extra = expansion.get("expansion_terms") or []
    if not extra:
        return query
    return query + " " + " ".join(str(x) for x in extra)


def validate_statutes(terms: Sequence[LegalTerm], valid_uids: Iterable[str]) -> list[str]:
    """Return ``(term_id, sec_uid)`` pairs that point at non-existent sections."""
    valid = set(valid_uids)
    return [
        f"{t.id}:{uid}" for t in terms for uid in t.statutes if uid not in valid
    ]
