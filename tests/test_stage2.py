"""Unit tests for Stage 2 (cleaning, normalisation, segmentation, chunking).

Run with ``python -m pytest tests/test_stage2.py -q``.  The tests deliberately
avoid the Hugging Face tokenizer so they run offline; chunk budgeting is tested
with ``WhitespaceTokenizer``, where one word is one token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import textnorm as T
from src.bn_nlp import (
    BN_STOPWORDS,
    EN_STOPWORDS,
    WhitespaceTokenizer,
    bm25_terms,
    stem_bn,
    stem_en,
    word_tokens,
)
from src.lexicon import expand_query, load_lexicon, validate_statutes
from src.stage2 import chunk_section, make_header, process_section

ROOT = Path(__file__).resolve().parents[1]
LEXICON = ROOT / "data" / "lexicon" / "legal_terms.json"
SECTIONS = ROOT / "data" / "clean" / "sections.jsonl"


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------


def test_clean_strips_section_number_and_glued_footnote():
    text = "302. Whoever commits murder 3[or abets it] shall be punished."
    out, counts = T.clean_text(text, section_no="302")
    assert out.startswith("Whoever commits murder")
    assert "3[" not in out
    assert "[or abets it]" in out
    assert counts["section_numbers"] == 1
    assert counts["footnote_markers"] == 1


def test_clean_keeps_amendment_and_year_brackets():
    text = "21. Nothing in [the said Act] applies to [2012]."
    out, _ = T.clean_text(text, section_no="21")
    assert "[the said Act]" in out
    assert "[2012]" in out
    assert "[183]" in T.clean_text("21. See sections 177 to [183].", section_no="21")[0]


def test_clean_strips_variant_danda_and_combined_section_numbers():
    out, _ = T.clean_text("৪৭৷ (১) এই আইন।", section_no="47")
    assert out.startswith("(১)")
    out, _ = T.clean_text("38-40. [Repealed by section 14.]", section_no="38")
    assert out.startswith("[Repealed")
    out, _ = T.clean_text("23 and 24. [Repealed.]", section_no="23")
    assert out.startswith("[Repealed")
    out, _ = T.clean_text("205CC.(1) When in a case.", section_no="205CC")
    assert out.startswith("(1)")


def test_clean_removes_leading_bracket_footnote_only():
    out, counts = T.clean_text("4. [১] মাদকদ্রব্য নিয়ন্ত্রণ।", section_no="4")
    assert out.startswith("মাদকদ্রব্য")
    assert counts["footnote_markers"] == 1


def test_clean_removes_html_page_furniture_and_running_head():
    text = (
        "Government of the People's Republic of Bangladesh\n"
        "Page 12\n"
        "<p>Hello &amp; bye</p>\n"
        "- 13 -\n"
        "End."
    )
    out, counts = T.clean_text(
        text, running_heads=("Government of the People's Republic of Bangladesh",)
    )
    assert "<p>" not in out and "</p>" not in out
    assert "&amp;" not in out
    assert "Page 12" not in out
    assert "- 13 -" not in out
    assert "Hello & bye" in out
    assert counts["html_entities"] >= 1
    assert counts["html_tags"] == 2
    assert counts["page_number_lines"] >= 1
    assert counts["page_number_dashes"] >= 1
    assert counts["running_headers"] == 1


def test_clean_removes_omitted_passage_marker():
    out, counts = T.clean_text("The Act [* * *] applies.", section_no="1")
    assert "[* * *]" not in out
    assert counts["omitted_passages"] == 1


# ---------------------------------------------------------------------------
# Unicode normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decomposed,precomposed",
    [
        ("\u09A1\u09BC", "\u09DC"),  # ড + nukta -> ড়
        ("\u09A2\u09BC", "\u09DD"),  # ঢ + nukta -> ঢ়
        ("\u09AF\u09BC", "\u09DF"),  # য + nukta -> য়
    ],
)
def test_nukta_forms_normalise_to_precomposed(decomposed, precomposed):
    assert T.normalize_text(decomposed)[0] == precomposed
    assert T.normalize_text(precomposed)[0] == precomposed
    assert T.is_canonical(precomposed)


def test_normalise_digits_and_danda():
    out, counts = T.normalize_text("ধারা ৩০২৷")
    assert "302" in out
    assert "৩" not in out
    assert out.endswith("\u0964")  # Devanagari danda used by the splitter
    assert counts["bn_digits"] == 3
    assert counts["danda_variants"] == 1


def test_normalise_is_idempotent():
    for text in ("ড়", "য়", "৩০২", "কো", "কৌ", "ধারা ১২৷"):
        once, _ = T.normalize_text(text)
        twice, _ = T.normalize_text(once)
        assert once == twice


# ---------------------------------------------------------------------------
# language identification
# ---------------------------------------------------------------------------


def test_detect_lang():
    assert T.detect_lang("এই আইন প্রযোজ্য")[0] == "bn"
    assert T.detect_lang("This Act applies")[0] == "en"
    assert T.detect_lang("এই Act applies")[0] == "mixed"
    assert T.detect_lang("(1) 1973")[0] == "neutral"
    assert T.detect_lang("(1)") == ("neutral", 0.0)


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------


def test_sentence_split_bangla_danda():
    text = "(1) এই আইন প্রযোজ্য হইবে। (2) ইহা কার্যকর হইবে।"
    spans = T.split_sentences(text)
    assert [s[0] for s in spans] == ["(1) এই আইন প্রযোজ্য হইবে।", "(2) ইহা কার্যকর হইবে।"]
    assert spans[0][1] == 0
    assert text[spans[1][1] : spans[1][2]] == spans[1][0]


def test_sentence_split_keeps_closing_bracket():
    text = "(1) এক। [(2) দুই।] (3) তিন।"
    spans = T.split_sentences(text)
    assert [s[0] for s in spans] == ["(1) এক।", "[(2) দুই।]", "(3) তিন।"]


def test_sentence_split_english_abbreviation_guard():
    text = "No. 5 of 1898 was amended. The Court may act."
    spans = T.split_sentences(text)
    assert len(spans) == 2
    assert spans[0][0].endswith("was amended.")
    assert spans[1][0] == "The Court may act."


def test_clause_split_retains_markers_and_proviso():
    clauses = T.split_clauses("(1) প্রথম। (2) দ্বিতীয়, তবে শর্ত থাকে যে তৃতীয়।")
    assert [c["marker"] for c in clauses] == ["1", "2", "2"]
    assert [c["type"] for c in clauses] == ["sub-section", "sub-section", "proviso"]
    assert clauses[0]["text"].startswith("(1)")
    assert clauses[1]["text"].startswith("(2)")
    assert clauses[2]["text"].startswith("তবে")


def test_clause_split_bangla_letter_markers():
    clauses = T.split_clauses("(ক) প্রথম; (খ) দ্বিতীয়")
    assert [c["marker"] for c in clauses] == ["ক", "খ"]
    assert all(c["type"] == "paragraph" for c in clauses)


def test_split_all_offsets_round_trip():
    text = "(1) এক। [(2) দুই।] (3) তিন।"
    for sentence in T.split_all(text):
        assert text[sentence["start"] : sentence["end"]] == sentence["text"]
        for clause in sentence["clauses"]:
            assert text[clause["start"] : clause["end"]] == clause["text"]


def test_bracket_clause_keeps_bracket_pair():
    clauses = T.split_clauses("[(2) দুই।]")
    assert clauses[0]["text"] == "[(2) দুই।]"
    assert clauses[0]["marker"] == "2"


# ---------------------------------------------------------------------------
# tokenisation and BM25 view
# ---------------------------------------------------------------------------


def test_word_tokens_does_not_split_bangla_conjuncts():
    assert word_tokens("বাংলা, আইন।") == ["বাংলা", "আইন"]


def test_bm25_removes_stopwords_but_dense_text_does_not():
    text = "এই আইনের অধীনে এবং অভিযুক্তকে জামিন দেওয়া হইবে"
    terms = bm25_terms(text)
    assert "এই" not in terms
    assert "এবং" not in terms
    assert "আইন" in terms  # আইনের -> আইন
    assert "জামিন" in terms
    # The dense view is untouched: the same function is not applied to it.
    dense, _ = T.normalize_text(text)
    assert "এই" in dense and "এবং" in dense


def test_stemmers_are_conservative():
    assert stem_bn("আইনের") == "আইন"
    assert stem_bn("অভিযুক্তকে") == "অভিযুক্ত"
    assert stem_bn("বিষয়") == "বিষয়"
    assert stem_bn("ঘর") == "ঘর"
    assert stem_en("offences") == "offence"
    assert stem_en("bails") == "bail"
    assert stem_en("accused") == "accused"


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def _fake_section(
    n_sentences: int = 12, words_per_sentence: int = 20
) -> tuple[dict, str, list[str], list[tuple[int, int]]]:
    sentences = [
        f"Sentence {i} " + " ".join(f"w{i}_{j}" for j in range(words_per_sentence))
        for i in range(n_sentences)
    ]
    text = " ".join(sentences)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for sent in sentences:
        start = text.find(sent, cursor)
        offsets.append((start, start + len(sent)))
        cursor = start + len(sent)
    sec = {
        "sec_uid": "test_s1",
        "act_id": 1,
        "act_name": "Test Act",
        "act_name_bn": "পরীক্ষা আইন",
        "section_no": "1",
        "title": "Test section",
        "text_en": text,
    }
    return sec, text, sentences, offsets


def test_chunking_respects_budget_and_overlap():
    tok = WhitespaceTokenizer()
    sec, text, sentences, offsets = _fake_section()
    tokens = [tok.count(s) for s in sentences]
    chunks = chunk_section(sec, "en", sentences, offsets, tokens, text, tok, max_tokens=120, overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["n_tokens"] <= 120
        assert text[chunk["char_start"] : chunk["char_end"]] == chunk["body"]
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur["overlap_ok"]
        if cur["overlap_tokens"]:
            overlap = text[cur["char_start"] : prev["char_end"]]
            assert prev["body"].endswith(overlap)
            assert cur["body"].startswith(overlap)
            assert cur["overlap_tokens"] <= 10


def test_chunking_never_splits_a_unit_larger_than_budget():
    tok = WhitespaceTokenizer()
    # One very long sentence must be cut on word boundaries, not overflow.
    sentence = " ".join(f"w{i}" for i in range(150))
    sec = {
        "sec_uid": "test_s2",
        "act_id": 1,
        "act_name": "Test Act",
        "section_no": "2",
        "title": "Long sentence",
    }
    chunks = chunk_section(sec, "en", [sentence], [(0, len(sentence))], [150], sentence, tok, max_tokens=100, overlap=10)
    assert len(chunks) > 1
    assert all(c["n_tokens"] <= 100 for c in chunks)
    # Every word survives in the concatenated bodies.
    joined = " ".join(c["body"] for c in chunks)
    assert all(f"w{i}" in joined for i in range(150))


def test_process_section_preserves_sec_uid_and_separates_bm25():
    tok = WhitespaceTokenizer()
    sec, _, _, _ = _fake_section()
    row, segments, chunks, stats = process_section(sec, tok, "test", max_tokens=120, overlap=10)

    assert row["sec_uid"] == "test_s1"
    assert row["lang"] == "en"
    assert chunks and all(c["sec_uid"] == "test_s1" for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert all(c["n_tokens"] <= 120 for c in chunks)
    assert stats["chunks"] == len(chunks)
    assert segments and all(s["sec_uid"] == "test_s1" for s in segments)
    # BM25 view drops "the"/"and"; dense view keeps them.
    sec2 = dict(sec, text_en="(1) The accused and the informant shall appear. (2) The Court may act.")
    row2, _, _, _ = process_section(sec2, tok, "test", max_tokens=200, overlap=10)
    assert "the" in row2["text_norm"].lower()
    assert "the" not in row2["text_bm25"].split()


def test_make_header_is_citable_in_both_languages():
    sec = {"act_name": "Penal Code, 1860", "act_name_bn": "দণ্ডবিধি, ১৮৬০", "section_no": "302", "title": "Punishment for murder"}
    assert "s. 302" in make_header(sec, "en")
    assert "ধারা 302" in make_header(sec, "bn")


# ---------------------------------------------------------------------------
# lexicon
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def terms():
    assert LEXICON.exists(), "legal_terms.json is an authored Stage 2 input"
    return load_lexicon(LEXICON)


def test_lexicon_size_and_shape(terms):
    assert 40 <= len(terms) <= 80
    ids = [t.id for t in terms]
    assert len(ids) == len(set(ids))
    for term in terms:
        assert term.en and term.bn
        assert term.label_en and term.label_bn


@pytest.mark.parametrize(
    "query,concept",
    [
        ("jamin lagbe", "bail"),
        ("naraji petition kivabe korbo", "naraji"),
        ("এজাহার করতে দেরি হলে", "fir"),
        ("how long can police keep someone on remand", "remand"),
        ("cognizable offence e police ki korte pare", "cognizable"),
    ],
)
def test_query_expansion_matches_colloquial_and_formal_forms(terms, query, concept):
    result = expand_query(query, terms)
    assert concept in {c["id"] for c in result["concepts"]}
    assert result["expansion_terms"], "expansion should add cross-lingual terms"


def test_expansion_adds_cross_lingual_synonyms(terms):
    result = expand_query("jamin", terms)
    assert any("জামিন" in form for form in result["expansion_terms"])
    assert any(form.lower() in {"bail", "release on bail", "bail bond"} for form in result["expansion_terms"])


def test_statute_references_resolve(terms):
    if not SECTIONS.exists():
        pytest.skip("sections.jsonl not built")
    with SECTIONS.open(encoding="utf-8") as fh:
        uids = [json.loads(line)["sec_uid"] for line in fh if line.strip()]
    assert validate_statutes(terms, uids) == []
