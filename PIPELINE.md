# How the legal QA application works

This document describes the current code path from a user's question to the displayed result. It documents the implementation as it exists; it does not describe a planned Gemini or BERT integration.

## End-to-end flow

```text
User types a question
        |
        v
Streamlit app validates non-empty input
        |
        v
Load cached Retriever (BM25 only in the demo)
        |
        v
Normalize/expand legal terms, tokenize query, calculate BM25 scores
        |
        v
Apply exact section-number promotion and special threat-query rule
        |
        v
Keep up to five unique section results
        |
        +--> Display top passage excerpt
        +--> Display retrieved sections and original source text
```

## 1. Preparing the legal corpus (offline)

Before a user searches, the repository's collection and preprocessing pipeline prepares statutory data:

1. `src/ingest.py` collects the selected act sources and writes canonical section records under `data/clean/` (notably `sections.jsonl`).
2. `src/stage2.py` and its helpers in `src/textnorm.py`, `src/bn_nlp.py`, `src/preprocess.py`, and `src/lexicon.py` normalize text, detect script, segment text, and create searchable chunks. The outputs include `data/clean/chunks.jsonl` and the legal dictionary `data/lexicon/legal_terms.json`.
3. `src/index.py` builds the persistent search artifacts in `indexes/`: `chunks.jsonl`, `bm25_tokens.pkl`, and optionally dense vectors/FAISS files. Each retrieval chunk keeps its section ID and metadata (act, section, language, title, and text).

The current corpus has 2,970 sections from 38 acts and 3,584 chunks. English and Bangla are not paired for every act; the app presents the source text in its original language.

## 2. User submits a question

The Streamlit interface is `app.py`. The user enters a question in the search box and presses **Search laws**. The app rejects an empty/whitespace-only question. It does not ask a BERT model to classify or rewrite the question.

The app calls `load_retriever()`, which is cached by Streamlit so the index is loaded once per running app process. It explicitly constructs `Retriever(dense=False)`, so this demo does not load BGE-M3 or use the saved dense vectors. Although the call passes `system="hybrid_expanded"`, dense retrieval is disabled, so the effective ranking is BM25 plus the application's query-expansion and special-case rules.

## 3. Query expansion and BM25 search

`src/retrieve.py` implements `Retriever.search(question, k=5, system=..., expand=True)`:

1. It starts with the original question.
2. It checks for a limited set of threat/intimidation terms. Those queries receive additional Penal Code terminology and may prioritize Penal Code sections 503/506 (and 507 for anonymous threats).
3. It loads the hand-built legal lexicon and matches English, Bangla, and colloquial/romanized Bangla terms. `src/lexicon.py` adds cross-language synonyms and related legal terms to the query. The user's original words are retained.
4. `src/index.py`'s `tokens()` function tokenizes the query into case-folded word tokens, including Bangla characters.
5. If `rank_bm25` is installed, it scores each chunk with `BM25Okapi`. Otherwise, `retrieve.py` computes BM25-style scores with term frequency, document frequency, and document-length normalization.
6. If the question contains an explicit reference such as “section 302” or “ধারা ৩০২”, the matching section is promoted to the start of the results.
7. The retriever removes duplicate sections (a long provision may have multiple chunks), then returns up to five results. Each result includes the chunk and a numeric BM25 score.

The lexicon and special threat handling are code-based query processing; they are not a general natural-language understanding model. Current search ranking is lexical BM25 in the demo. BGE-M3 dense search is available in the retrieval code for experiments when enabled and model weights are available, but is not used by the Streamlit demo.

## 4. Answer construction in the current demo

`app.py` passes the returned hits to `extractive_answer()` in `src/generate.py`. This is not an LLM and does not synthesize an answer from multiple sections. It takes the first (highest-ranked) result, displays up to the first 1,000 characters of its text, and appends a fixed citation in the form `[Act name §section]`. If no result or no text is available, it returns “Not found in the provided law.”

The app labels this content as a **source excerpt** / **top source passage** and separately shows the retrieved provisions in expandable panels. It escapes the excerpt before rendering HTML. It also displays a legal-information-not-legal-advice notice.

## 5. Procedure lookup and citation verification status

`src/qa.py` contains a separate `LegalQA` orchestration path. It retrieves hits, may attempt to attach a procedure row for an arrest/warrant/bail question when the top hit is a Penal Code provision, produces the same extractive answer, then calls `src/verify.py` to check that cited act/section pairs exist and appeared among the retrieved hits.

The current Streamlit `app.py` does **not** call `LegalQA`; therefore it does not run that procedure join or citation checker. Also, `data/clean/procedure.json` currently has zero rows, so there are no procedure facts to attach even through `LegalQA`. `src/verify.py` checks citation existence and retrieval membership only; it does not decide whether a provision legally supports the answer.

## 6. What is and is not happening

| Component | Current behavior |
|---|---|
| BERT question classifier | Not present |
| Legal term understanding | Hand-built dictionary expansion plus a few explicit rules |
| Demo retriever | BM25 over local statute chunks |
| Dense model | BGE-M3 artifacts/code available for research; disabled in demo |
| Generative model / Gemini | Not connected |
| Displayed “answer” | Excerpt from the top-ranked section with a fixed citation |
| Procedure card | Not shown in current app; procedure dataset is empty |
| Citation verification | Available through `src/qa.py`, but not called from current app |
| Legal notice | Shown in sidebar and below results |

## 7. Source files and useful commands

- UI and request handling: `app.py`
- Retrieval: `src/retrieve.py`
- Query expansion: `src/lexicon.py`
- Offline index build: `src/index.py`
- Extractive display text and unused prompt template: `src/generate.py`
- Optional QA orchestration and procedure join: `src/qa.py`
- Deterministic citation check: `src/verify.py`
- Corpus paths and model configuration: `src/config.py`

Build a BM25-only index with `python -m src.index --bm25-only`, then launch the UI with `streamlit run app.py`.

## 8. Example request trace

For `What does section 302 cover?`:

1. The app receives the text and checks it is not blank.
2. The cached retriever loads the chunk and token indexes; dense mode remains off.
3. The exact-section regular expression detects `302` and promotes matching section 302 chunks.
4. BM25 ranks the chunks; the retriever returns unique sections, up to five.
5. The app displays up to 1,000 characters from the first hit and cites its act and section.
6. The app lists each retrieved source passage for inspection. No model generates a paraphrased legal conclusion.

This prototype is a legal-statute search aid. Users should inspect the source passages and confirm the current official text; the app is not legal advice.
