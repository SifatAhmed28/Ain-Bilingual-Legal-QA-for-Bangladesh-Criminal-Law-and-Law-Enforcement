# Project status and continuation handoff

Last updated: 2026-11-27  
Project: **Bilingual Legal QA for Bangladesh Criminal Law and Law Enforcement**  
Working title: **Bangla-English Hybrid Retrieval and Citation-Verified QA for Bangladesh Criminal Procedure and Policing**

This is the restart point for the next work session. The original project specification is in [AGENTS.md](AGENTS.md). Stages 1 and 2 predate this session; the code/data outputs from this session cover a functional baseline for later stages, while research artifacts still needing source validation or human review are explicitly marked as incomplete.

## Research framing

Problem: Citizens and junior officers have difficulty finding which law applies to a situation, in Bangla or English.

- **RQ1:** Which retrieval method performs best for Bangla, English, and colloquial Bangla legal queries: BM25, multilingual dense, or hybrid?
- **RQ2:** Does structured procedure lookup for arrest, bail, and court reduce errors compared with LLM-only answers?
- **RQ3:** Does citation verification reduce hallucinated section references?

The corpus currently covers 38 acts (criminal law, procedure, evidence, policing and related statutes), rather than a narrow five-act sample. It is not bilingual per-act: English and Bangla texts are linked by `sec_uid` when available, but the current source collection has zero fully bilingual acts. No machine translations were created.

## Existing work before this session

### Stage 1: data collection — completed

Implementation: `src/ingest.py`, `src/preprocess.py`, shared settings in `src/config.py`.

Main outputs:

- `data/clean/sections.jsonl`: 2,970 section records across 38 included acts.
- `data/clean/acts.jsonl`, `data/clean/dropped.jsonl`, `data/clean/coverage.csv`.
- Raw source snapshots under `data/raw/acts/` and `data/raw/bdlaws/`; source corpora under `BLAD/` and `UKIL-DB-EN/`.
- Stage report: `report/stage1_report.md`.

The report records 1,892 English-only sections, 1,078 Bangla-only sections, and no fully bilingual acts. Six repealed acts and 22 repealed/omitted sections are excluded from the canonical section output. Latest amendment date/year and source metadata are retained where found. One consequential caution: the data includes acts dated 2025–2026; these should be checked against official current law before any legal-facing use or publication.

### Stage 2: preprocessing — completed

Implementation: `src/stage2.py`, `src/textnorm.py`, `src/bn_nlp.py`, `src/lexicon.py`.

Main outputs:

- `data/clean/sections_norm.jsonl`, `data/clean/segments.jsonl`, `data/clean/chunks.jsonl`.
- `data/lexicon/legal_terms.json` (80 concepts, 589 surface forms).
- `results/stage2_token_stats.csv`, `results/stage2_summary.json`.
- Main report: `report/stage2_report.md`. The temporary smoke report and generated smoke outputs are retained under `archive_unnecessary/smoke_artifacts/`.

Full-corpus report: 2,970 sections, 5,945 sentences, 11,571 clauses, 3,584 chunks, 352 multi-chunk sections, 343,472 whitespace tokens and 574,762 subword tokens. Normalization applies NFC plus Bangla nukta recomposition, Bengali digit conversion for retrieval text, Unicode language detection, clause/sentence segmentation, BM25-only stopword/stemming and 400-token chunks with 50-token overlap. Dense text is not stopword-filtered.

## Work completed in this session

### Stage 3: procedure table — automatic extraction path added; local table still empty

- Added `src/procedure.py`; running it writes `data/clean/procedure.json`.
- It is intentionally conservative: unknown fields remain `null`, and extracted rows are marked `verified: false`.
- Current `data/clean/procedure.json` has **0 rows**. The parsed `sections.jsonl` treats CrPC content as section records; the cached CrPC HTML does not provide the Second Schedule as clean table rows.
- Added `src/procedure_auto.py`. It downloads the official Schedule II PDF, attempts table extraction, stores page/raw-cell provenance, and overlays the schedule changes listed in section 28 of the 2026 CrPC Amendment Act. It marks all rows unverified and flags grouped/ambiguous rows. It has not been run against the online PDF yet.
- Current authoritative source links: [CrPC Schedule II PDF](https://bdlaws.minlaw.gov.bd/upload/act/2025-08-17-15-16-07-75___170825-Schedule.pdf), [CrPC Amendment Act 2026, section 28](https://bdlaws.minlaw.gov.bd/act-print-1640/section-print-58332.html). Do not treat automatic extraction as legal verification.

### Stage 4: benchmark — 150-item seed created, not a final benchmark

- Added `src/benchmark.py`; generated `data/benchmark/questions.jsonl` (150 rows, 50 Penal Code/CrPC section title seeds × 3 language-style templates).
- Rows carry a `gold_sections` section ID and a deterministic 30/70 dev/test split.
- Every row is marked `review_status: needs_manual_review`. The Bangla templates use the source title as-is and may not be idiomatic Bangla. Review each question and gold section, replace weak prompts, add punishment/procedure/bail/evidence/scenario and 20 out-of-scope questions, then obtain a second annotator's 20–30 judgments and calculate agreement. Do not report this seed as validated test data.

### Stage 5: indexing — BM25 and dense index artifacts built

- Added `src/index.py` and `indexes/`.
- Built portable BM25 token corpus for **3,584** chunks: `indexes/chunks.jsonl`, `indexes/bm25_tokens.pkl`, `indexes/manifest.json`.
- `rank_bm25` is not installed, so retrieval includes a pure Python BM25 fallback. The fallback computes term document frequency at query time and is acceptable for a small experimental corpus, but should be optimized before scaling.
- Dense model: `BAAI/bge-m3` (matching Stage 2's tokenization report). Saved `indexes/dense_embeddings.npy` and `indexes/dense.faiss` are present (3,584 vectors). The model weights were deleted from the local cache, so dense query-time encoding requires downloading them again. The demo is configured BM25-only.

### Stages 6–9: retrieval, procedure join, answer and citation checks — local baseline

- Added `src/retrieve.py`: BM25, optional dense similarity, RRF, legal dictionary expansion and exact section-number promotion. Methods exposed through `Retriever.search(query, k=5, system=...)`.
- Added `src/qa.py`: end-to-end local pipeline with procedure attachment conditional on Penal Code top hit plus arrest/bail intent.
- Added `src/generate.py`: strict grounded-answer prompt builder and an extractive offline fallback. No generative LLM is connected; no API token was available in this environment.
- Added `src/verify.py`: checks fixed-format citations against known section ids and retrieved context. This is a deterministic existence/context check, not semantic entailment verification; no retry/regeneration orchestration or NLI judge is connected.
- Since the procedure table has 0 rows, the procedure card will currently be empty. Current extractive fallback returns source text and a fixed citation; it is a prototype, not a robust bilingual QA system.

### Stage 10: evaluation — metric code only

- Added `src/evaluate.py` with Recall@1/3/5, MRR and nDCG@5 grouped by language style.
- No system scores or generation evaluation were produced. The benchmark seed needs review first. Still needed: all S1–S5 runs, citation validity, manual/LLM faithfulness, procedure accuracy, out-of-scope abstention, ablations and 30–50 categorized failures with worked examples.

### Stage 11: demo — starter Streamlit app

- Added root `app.py`, using BM25 to search the local index and showing source excerpts, retrieved sections, and a legal-information-not-advice disclaimer. It requires indexes to exist; answer-language control and procedure card are not implemented.
- Follow-up UI fix: app is explicitly BM25-only (`Retriever(dense=False)`); it no longer loads BGE-M3 weights. Cached model weight files were deleted at the user's request, while saved vectors/FAISS index remain for reference. App language selector was removed because it had no effect; UI explains that source passages stay in their published language. Streamlit server and HTTP page response were checked successfully.
- Threat-query retrieval fix: common threat terms (including the typo `thread`) and Bangla threat terms prioritize Penal Code sections 503 and 506; anonymous-threat wording also adds section 507. Search now returns unique sections, and for this intent it suppresses unrelated lexical matches. The reported query now returns sections 503 and 506. Python syntax compilation and this focused query check passed.

### Stage 12: showcase report — assembled from completed work

Added `report/showcase_report.md` and a root `README.md` describing completed work without claiming unrun experiments. A full experimental course report still needs reviewed benchmark labels, agreement, retrieval/generation results, and error analysis.

## Added/changed files this session

- Added: `src/index.py`, `src/retrieve.py`, `src/procedure.py`, `src/benchmark.py`, `src/generate.py`, `src/verify.py`, `src/evaluate.py`, `src/qa.py`, root `app.py`.
- Added generated: `data/clean/procedure.json` (empty extracted table, source limitation noted), `data/benchmark/questions.jsonl` (150 review-required items), `indexes/chunks.jsonl`, `indexes/bm25_tokens.pkl`, `indexes/manifest.json`.
- Fixed: `src/retrieve.py` uses `expanded_query_text` correctly rather than treating the lexicon expansion result object as a string.
- Showcase organization added: root `README.md`, `requirements.txt`, `report/showcase_report.md`, and `archive_unnecessary/` for caches, smoke artifacts, and the old session log. Important source/data/code/report folders remain in their original locations.
- Showcase organization added: root `README.md`, `requirements.txt`, `report/showcase_report.md`, and `archive_unnecessary/` for caches, smoke artifacts, and the old session log. Important source/data/code/report folders remain in their original locations.

## Environment and checks

- Python 3.12. Installed in the local environment: NumPy, FAISS, sentence-transformers, Streamlit, scikit-learn, PyMuPDF, and requests. Missing: `rank_bm25`; the BM25 fallback is enabled.
- Checked credential presence without reading or printing secret values: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` were all unset. Configure a provider token only if a generation API is intentionally selected; never commit it.
- BM25 search and a threat-query retrieval case were checked; Streamlit served HTTP 200 in a prior UI check. Python syntax checks passed for local scripts. The online PDF extraction has not been executed. No unit-test suite was run in this session.
- Workspace is not a Git repository (`git status` reports “not a git repository”), so there is no commit history/status to reference.

## Suggested next-session order

1. Inspect and correct the retrieval/index code, then run a small retrieval smoke check; install `rank_bm25` only if desired, since fallback exists.
2. Source the official current CrPC Second Schedule, encode/parse procedure rows with provenance and independent legal review; do not silently use `null` as a fact.
3. Manually revise the benchmark seed, add missing question types and 20 out-of-scope examples, preserve held-out test split, and collect independent labels.
4. Build dense vectors using an approved/cached multilingual model and run S1–S5 retrieval evaluation on dev then test once frozen.
5. Add procedure display, language-aware generation, citation retry/abstention and semantic support checks, then evaluate and complete the report.
6. Launch the Streamlit demo only after the local environment and its legal content have been reviewed.

## Commands

```powershell
python -m src.index --bm25-only
python -m src.index                 # downloads/loads BGE-M3 for dense vectors
python -m src.procedure_auto        # needs Internet and PyMuPDF
python -m src.benchmark
streamlit run app.py
```


## 2026-11-27: BERT-style understanding and Gemini answer generation

The proposed architecture is compatible with the existing project:

1. **Understand and retrieve:** detect the question language/style, normalize Bangla digits, expand terms with `data/lexicon/legal_terms.json`, and retrieve relevant statute sections with the existing BM25/dense/hybrid retriever. BGE-M3 is already the project's multilingual dense retrieval encoder; it is an encoder in the BERT family of retrieval approaches and is a better fit for cross-lingual semantic retrieval than plain BERT. The dense weights are not currently cached locally, so the app baseline remains BM25 unless those weights are downloaded.
2. **Answer:** send the question plus only the retrieved legal excerpts (and a verified procedure row when available) to Gemini to draft a clear, structured answer in the user's language. Gemini should format and explain the retrieved evidence, not search the whole law independently or invent legal facts.
3. **Verify:** run citation verification after generation. Citations must match known section identifiers and the retrieved context; if verification fails, retry once with the error or abstain. The current `src/qa.py` uses an extractive fallback and deterministic citation checking, so the Gemini call and retry are not implemented yet.

**Model choice guidance:** BERT is not itself a question-answering database or answer generator. A base BERT model can classify intent (definition, punishment, arrest, bail, evidence, scenario) if trained and evaluated on labeled examples, but this project has no reviewed intent dataset yet. For now use the existing retriever plus explicit intent rules; compare a multilingual dense encoder against BM25/hybrid in the benchmark. This keeps the retrieval experiment measurable and avoids adding an unsupported BERT classifier.

**Gemini API key:** store the key in an environment variable such as `GEMINI_API_KEY` or in a local `.env` file excluded from version control; never put it in source code, Streamlit UI, or this status file. Keep API access server-side. Before sending prompts, avoid personal identifiers and tell users not to submit sensitive case details. Google's current pricing page says some Gemini API models have a free tier with limited access and quotas; availability/rate limits can vary, and free-tier submitted content may be used to improve Google products. Recheck the current Google AI Studio terms and quotas before deployment. Source: https://ai.google.dev/gemini-api/docs/pricing

**Current integration status:** no Gemini key is present in this environment (checked without printing any secret value). No Gemini integration has been made in this session. Existing runnable path remains local retrieval + extractive answer.

**Next implementation steps:**

- Add a Gemini provider function in `src/generate.py` that accepts question, retrieved context and procedure facts; configure model name, low temperature, output cap, timeout, and friendly quota/error handling.
- Add optional API configuration via environment only; add `.env` to `.gitignore` if a dotenv workflow is used. Never commit a real key.
- Update `src/qa.py` to call Gemini after retrieval, then verify the citations and retry once or abstain. Preserve extractive mode when no key is configured.
- Update `app.py` to show the structured answer alongside retrieved sources, and label the answer as generated from those sources.
- Evaluate this generator against the existing local fallback using reviewed questions, citation validity, faithfulness checks and latency/quota observations before making claims about quality.

## Current application request flow

A detailed, code-accurate explanation of corpus preparation, user query handling, retrieval, excerpt construction, and current limitations is documented in [PIPELINE.md](PIPELINE.md). Key correction for the next session: the Streamlit app calls `Retriever(dense=False)` and `extractive_answer()` directly. It does not currently invoke `LegalQA`, the procedure join, citation verification, a BERT model, or Gemini. Treat the displayed result as a retrieved source excerpt, not a generated legal answer.

## 2026-11-27: Streamlit UI refinement

Updated `app.py` while keeping the local BM25 search pipeline unchanged. The page now uses a calmer compact header, a more prominent search form, one-click example queries, persisted search results, clearer empty/error states, a highlighted best matching provision, expandable remaining results, and a full-passage expander for the top match. Copy and labels clarify that the app shows original legal text rather than a generated legal conclusion. The legal-information disclaimer remains visible. Dense retrieval remains disabled in the demo; no BERT or Gemini integration was added.

No test suite or runtime verification was run.
