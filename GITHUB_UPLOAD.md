# GitHub upload preparation

This repository is prepared as a lightweight, runnable BM25 search demo and NLP research codebase. It does not bundle the downloaded BLAD/UKIL corpora, raw source snapshots, full cleaned intermediate corpus, or dense model vectors.

## Included

- Streamlit demo: `app.py`
- Application and data pipeline code: `src/`
- BM25 runtime files: `indexes/chunks.jsonl`, `indexes/bm25_tokens.pkl`, `indexes/manifest.json`
- Legal query lexicon: `data/lexicon/legal_terms.json`
- Benchmark seed: `data/benchmark/questions.jsonl` (generated questions; still requires manual review)
- Reports, project documentation, dependencies, and stage 2 tests

## Excluded by `.gitignore`

- Downloaded source datasets: `BLAD/`, `UKIL-DB-EN/`, `data/raw/`
- Cleaned intermediate files under `data/clean/`
- Dense vectors and FAISS index (large; not used by this BM25-only app)
- `archive_unnecessary/`, Python caches, virtual environments, and local secrets

The BM25 chunk index contains excerpts of statutory text and is required by the demo. Confirm that the source-data redistribution terms permit publishing this derived index before making the repository public. The app itself does not download a model or require an API key.

## Run after cloning

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The checked-in BM25 index makes the demo runnable without the excluded source corpora. Rebuilding the index from scratch requires restoring the local corpus files and running the ingestion, preprocessing, and indexing commands documented in `README.md` and `PIPELINE.md`.

## Before publishing

1. Confirm the license and redistribution terms for source-derived index text.
2. Review benchmark questions and any reports for material intended to be public.
3. Create a GitHub repository, add it as the Git remote, and push the reviewed commit.

No API credentials belong in Git. Keep any local `.env` file untracked.
