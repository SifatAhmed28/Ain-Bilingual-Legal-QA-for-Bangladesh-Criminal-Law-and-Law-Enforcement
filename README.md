# Bilingual Legal QA for Bangladesh Criminal Law and Law Enforcement

**Working title:** Bangla-English Hybrid Retrieval and Citation-Verified QA for Bangladesh Criminal Procedure and Policing

An NLP course project exploring how citizens and junior officers can find relevant Bangladesh criminal-law provisions using English, Bangla, and colloquial queries. The current deliverable is a research prototype: a curated legal corpus, Bangla-aware preprocessing pipeline, BM25 retrieval, an optional dense/hybrid retrieval implementation, evaluation scaffolding, and a Streamlit search demo.

> **Scope notice:** the app returns source excerpts as a research aid. It is not a lawyer, does not provide legal advice, and must not be used to make urgent legal decisions. Verify provisions against current official sources.

## Research questions

1. Which retrieval method performs best for formal English, formal Bangla, and colloquial Bangla legal questions: BM25, multilingual dense retrieval, or hybrid retrieval?
2. Does structured procedure lookup for arrest, bail, and trial court reduce errors compared with language-model-only answers?
3. Does citation verification reduce unsupported or hallucinated section references?

## What is implemented

- **Data collection:** section-level legal records from bdlaws, BLAD, and UKIL-DB-EN, with source and act metadata.
- **Corpus:** 2,970 section records across 38 included acts. Current coverage is mostly monolingual by act: 1,892 sections have English text, 1,078 have Bangla text, and none of the included acts is fully bilingual. No machine translation was added.
- **Bangla-aware preprocessing:** Unicode NFC and Bangla nukta normalization, Bengali-digit normalization for retrieval, script-based language labels, sentence/clause segmentation, stopword/stemming view for BM25 only, and section-based overlapping chunks.
- **Lexicon:** 80 legal concepts and 589 English, Bangla, and colloquial forms for query expansion.
- **Corpus statistics:** 5,945 sentences, 11,571 clauses, 3,584 retrieval chunks, and recorded whitespace/subword token counts.
- **Retrieval:** BM25 baseline with exact-section promotion, legal-term expansion, a targeted threat-query mapping, and optional dense/RRF hybrid code.
- **Saved indexes:** BM25 token corpus, dense vectors, and a FAISS index are present under `indexes/`. The app intentionally uses BM25 only; BGE-M3 model weights were removed to avoid the multi-gigabyte runtime download. Dense query-time search needs the model weights again.
- **Prototype UI:** modern Streamlit app showing ranked original source excerpts and a legal-information disclaimer.
- **Procedure extraction utility:** local CPU-compatible best-effort extractor for the official CrPC Second Schedule, preserving source pages and raw table cells.

## NLP methods used

| Component | NLP method or research purpose |
|---|---|
| Text normalization | Unicode normalization, Bangla-specific character handling, digit normalization |
| Language handling | Bangla/Latin script detection and English/Bangla/colloquial query variants |
| Segmentation | Sentence, clause, and legal subsection splitting |
| Tokenization | Whitespace baseline vs model subword-tokenizer counts |
| Lexical retrieval | BM25 with stopwords and light stemming applied only to its token view |
| Semantic retrieval | Multilingual sentence embeddings and FAISS (optional runtime model download) |
| Hybrid retrieval | Reciprocal-rank fusion, exact-section boosting, and legal dictionary expansion |
| Evaluation | Recall@1/3/5, MRR, and nDCG@5 scaffolding grouped by language style |
| Citation checking | Deterministic check that citations exist and appeared in retrieved context |

## Run the demo

Python 3.10+ is recommended. From the project root:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The demo uses the existing local BM25 index. If the index needs rebuilding:

```powershell
python -m src.index --bm25-only
```

No GPU or large language model is needed for the local demo. For a low-configuration machine, keep the app on BM25 and do not build dense embeddings.

The app is a **retrieval demo**, not a generated-answer system: it shows legal source passages and does not translate or synthesize them. A question may return source text in a different language from the query because source acts are not fully bilingual.

## Local reproduction workflow

From the project root, rebuild the corpus/index on CPU using cached source snapshots:

```bash
python -m src.ingest --offline
python -m src.preprocess
python -m src.stage2 --tokenizer-model none
python -m src.index --bm25-only
streamlit run app.py
```

The `none` tokenizer mode avoids model downloads and GPU use; its subword counts are approximate. Stage 1 offline mode requires the raw snapshots already included in this project. To fetch procedure-table candidates, install PyMuPDF if needed and run `python -m src.procedure_auto`; this step needs Internet access to download the official schedule PDF, but does not need a GPU or model weights.

Automatic procedure-table rows preserve page/source evidence and remain unverified. Extraction is best-effort, not legal validation.

## Project structure

```text
data/                 raw sources, cleaned sections, chunks, benchmark, lexicon
src/                  ingestion, preprocessing, indexing, retrieval, QA, metrics
indexes/              BM25 tokens, dense vectors, FAISS index
results/              preprocessing statistics and summaries
report/                stage reports and current showcase report
BLAD/                  BLAD source corpus
UKIL-DB-EN/             UKIL English source corpus
archive_unnecessary/   retained caches, smoke artifacts, and session log
src/procedure_auto.py  local Schedule II candidate extractor
app.py                  Streamlit search prototype
AGENTS.md               original project pipeline specification
CURRENT_STATUS.md       detailed implementation handoff
```

## Research status and limitations

- Stages 1–2 are complete and documented in `report/stage1_report.md` and `report/stage2_report.md`.
- The benchmark contains 150 **generated seed questions**. They remain marked `needs_manual_review`; no retrieval scores should be reported as validated results from this seed.
- The current `data/clean/procedure.json` is empty. The local helper can automatically extract candidate rows from the official schedule PDF, but it has not yet been run against the online PDF and its output is marked unverified.
- Dense vectors are saved, but the BGE-M3 weights were removed from the local Hugging Face cache. The demo remains BM25-only.
- No dense-vs-BM25 evaluation, generation evaluation, annotator agreement, or final experimental findings are claimed.
- The collected set is a curated 38-act corpus, not every Bangladesh statute or case. Check source licenses and law currency before sharing or legal-facing use.

For complete stage history and next steps, see [CURRENT_STATUS.md](CURRENT_STATUS.md). For the polished project summary, see [report/showcase_report.md](report/showcase_report.md).
