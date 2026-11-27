"""Bangla/English tokenisation, stopwords and light stemming for BM25.

Two tokenisation views are kept deliberately separate:

``whitespace_tokens``
    ``text.split()`` — the unit a BM25 index would see if punctuation were not
    stripped.  It is the baseline in the Stage 2 token-count comparison.

``bm25_terms``
    Whitespace tokens with punctuation stripped, stopwords removed and a
    *light* stemmer applied.  This view is used **only** for BM25.  Dense
    embeddings get the unmodified normalised text, because subword models
    already handle inflection and stopwords in context.

``SubwordTokenizer``
    The embedding model's own tokenizer (default: the Stage 3 model).  It is
    used for chunk budgeting and for the Bangla-vs-English token-cost analysis.
"""

from __future__ import annotations

import math
import re
import string
from typing import Iterable, Sequence

from .textnorm import BN_LETTER_RE, LATIN_LETTER_RE, script_counts

# ---------------------------------------------------------------------------
# stopwords
# ---------------------------------------------------------------------------

# Small, hand-checked Bangla list.  Legal content words (আইন, ধারা, আদালত,
# জামিন, অভিযোগ ...) are intentionally absent: removing them would damage
# recall.  Multi-word expressions are not listed because matching is per token.
BN_STOPWORDS: frozenset[str] = frozenset(
    """
    অতএব অথচ অথবা অন্য অনেক আবার আর আরও ইতিমধ্যে ইত্যাদি ইহা ইহার এখন এখানে
    এবং এমন এমনকি এই এইসব একই একটি একজন এটা এটি এত এতে এদের এর এসব
    হয় হয়ে হওয়া হইতে হইবে হইয়া হইয়াছে হবে হচ্ছে হন ছিলেন ছিল আছে নাই নেই
    নয় নহে না করে করেন করবে করবেন করিবে করিতে করিয়া করা করার পারে পারিবে
    পারবেন জন্য জন্যে সহিত সাথে সঙ্গে মধ্যে দ্বারা কর্তৃক হতে থেকে উপর উপরে
    নিচে নিকট কাছে পর্যন্ত ব্যতীত ব্যতিত ছাড়া আগে পরে ক্ষেত্রে বিষয়ে সম্পর্কে
    অনুযায়ী মতে মতো পরিবর্তে সাপেক্ষে পক্ষে দিকে মাধ্যমে কারণে ফলে ভিতরে
    বাহিরে মাত্র প্রায় তখন সেখানে তবে যদি যদিও কারণ সুতরাং তথা কিংবা কিম্বা
    অর্থাৎ যেমন যথা প্রভৃতি বরং যেন যখন যেখানে যে যেই যা যাহা যিনি যাহার যাহাদের
    যার তার তাদের তাহা তাহার তাহাদের তাহলে তিনি তাঁকে তাঁর সে সেই তা তাই আমি
    আমার আমাদের আপনি আপনার আমরা নিজ নিজের সব সকল সমস্ত প্রত্যেক কিছু উভয়
    টি টা গুলি গুলো দের খানি খানা উক্ত অধীন ব্যতিরেকে
    """.split()
)

# Standard function words.  "provided" is deliberately kept: it signals a
# proviso and is a useful retrieval cue.
EN_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with without under over as is are was were be been being shall will
    may might must can could would should has have had do does did not no nor
    so such any all each every other same he she it they we you i his her its
    their our your him them us who which whom whose where when why how there
    herein hereof thereto whereas
    """.split()
)

# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------

# string.punctuation is ASCII-only; add the Bangla/typographic marks that occur
# at token edges.  Internal marks (hyphen, slash, dot in "s.302") are kept.
EDGE_PUNCT = string.punctuation + "।৷“”‘’—–…«»·"


def whitespace_tokens(text: str) -> list[str]:
    """Plain ``str.split`` — the baseline token count."""
    return text.split()


def word_tokens(text: str) -> list[str]:
    """Whitespace tokens with edge punctuation removed.

    A regex like ``\\w+`` cannot be used for Bangla: combining marks (Mn) break
    the match, so ``"বাংলা"`` would become four one-letter tokens.  Splitting on
    whitespace and trimming edges is script-neutral and keeps conjuncts intact.
    """
    out: list[str] = []
    for raw in text.split():
        token = raw.strip(EDGE_PUNCT)
        if token:
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# light stemmers
# ---------------------------------------------------------------------------

# Longest suffixes first.  Single-character suffixes are deliberately excluded:
# stripping "য়" from বিষয় or "র" from ঘর does more harm than good.
BN_SUFFIXES: tuple[str, ...] = (
    "গুলির",
    "গুলোর",
    "গুলোতে",
    "গুলিতে",
    "সমূহের",
    "দিগের",
    "দিগকে",
    "খানির",
    "খানার",
    "টিতে",
    "টাতে",
    "দেরকে",
    "গুলো",
    "গুলি",
    "সমূহ",
    "খানি",
    "খানা",
    "টির",
    "টার",
    "েরা",
    "হইয়া",
    "হইতে",
    "করিয়া",
    "করিতে",
    "করার",
    "হইবে",
    "হইয়াছে",
    "হয়েছে",
    "করিবেন",
    "করিবে",
    "করেন",
    "করেছেন",
    "ছিলেন",
    "ভাবে",
    "মূলক",
    "যোগ্য",
    "দের",
    "গুলি",
    "টা",
    "টি",
    "গণ",
    "ভাব",
    "ত্ব",
    "তা",
    "হয়ে",
    "হতে",
    "করে",
    "করা",
    "হবে",
    "হয়",
    "আছে",
    "নাই",
    "নেই",
    "ছিল",
    "ের",
    "য়ে",
    "কে",
    "তে",
    "রা",
)

# Roots that happen to end in a suffix string; never strip these.
BN_STEM_EXCEPTIONS: frozenset[str] = frozenset(
    "কর্তা বক্তা শ্রোতা দাতা গ্রহীতা পিতা মাতা সভা কথা ব্যাখ্যা প্রতিভা".split()
)

MIN_STEM = 3


def stem_bn(word: str) -> str:
    """Conservative suffix stripping for Bangla nominals and verb forms."""
    if len(word) <= MIN_STEM or word in BN_STEM_EXCEPTIONS:
        return word
    for suffix in BN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[: -len(suffix)]
    return word


def stem_en(word: str) -> str:
    """Plural-only English normalisation.

    Deliberately weaker than Porter: the legal register is noun-heavy and
    aggressive stemming turns "conviction" into "convict", conflating the
    offence and the verdict.
    """
    if len(word) <= 3:
        return word
    lower = word.lower()
    if lower.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if lower.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if lower.endswith("s") and not lower.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def bm25_terms(text: str, stopwords: Iterable[str] | None = None) -> list[str]:
    """Stopword-filtered, lightly stemmed terms for the BM25 index."""
    stop = stopwords if stopwords is not None else (BN_STOPWORDS | EN_STOPWORDS)
    out: list[str] = []
    for token in word_tokens(text):
        low = token.lower()
        if low in stop:
            continue
        if BN_LETTER_RE.search(low):
            low = stem_bn(low)
        elif low.isascii() and low.isalpha():
            low = stem_en(low)
        if len(low) >= 2 or low.isdigit():
            out.append(low)
    return out


def bm25_text(text: str, stopwords: Iterable[str] | None = None) -> str:
    return " ".join(bm25_terms(text, stopwords))


# ---------------------------------------------------------------------------
# subword tokenisation
# ---------------------------------------------------------------------------


class BaseTokenizer:
    name = "base"

    def count(self, text: str) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def count_batch(self, texts: Sequence[str]) -> list[int]:
        return [self.count(t) for t in texts]


class WhitespaceTokenizer(BaseTokenizer):
    name = "whitespace"

    def count(self, text: str) -> int:
        return len(text.split())


class ApproxTokenizer(BaseTokenizer):
    """Offline fallback: ~2 Bangla chars/token, ~4 Latin chars/token."""

    name = "approx(char-based)"

    def count(self, text: str) -> int:
        bn, latin = script_counts(text)
        other = max(0, len(text) - bn - latin)
        return max(1, math.ceil(bn / 2.0 + (latin + other) / 4.0))


class SubwordTokenizer(BaseTokenizer):
    """The embedding model's tokenizer (tokenizer files only, no weights)."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self.name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def count_batch(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encoded = self._tok(list(texts), add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]


def load_tokenizer(model_name: str | None) -> BaseTokenizer:
    """Load the subword tokenizer, falling back to an approximation offline."""
    if not model_name or model_name.lower() in {"none", "whitespace"}:
        return WhitespaceTokenizer()
    try:
        return SubwordTokenizer(model_name)
    except Exception as exc:  # pragma: no cover - depends on network/cache
        print(f"[warn] could not load tokenizer {model_name!r}: {exc}")
        print("[warn] falling back to char-based approximation")
        return ApproxTokenizer()
