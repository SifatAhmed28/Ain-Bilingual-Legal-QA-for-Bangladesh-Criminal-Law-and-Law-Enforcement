stage one and two is complete , now complete others session check the token before finish the token write a .md file for current status so that next time when i start the work again .md file have all details of past work, details: Project pipeline: Bilingual Legal QA for Bangladesh Criminal Law and Law Enforcement

Working title: Bangla-English Hybrid Retrieval and Citation-Verified QA for Bangladesh Criminal Procedure and Policing.

0. Project framing (write this first)
   Problem: Citizens and junior officers can't easily find which law applies to a situation, in Bangla or English.
   Research questions (these make it an NLP project, not just an app):
   RQ1: Which retrieval method works best for Bangla, English and colloquial Bangla legal queries (BM25, multilingual dense, hybrid)?
   RQ2: Does structured procedure lookup (arrest, bail, court) reduce errors compared with LLM-only answers?
   RQ3: Does citation verification reduce hallucinated sections?
   Scope: about 10–15 acts: Penal Code 1860, CrPC 1898, Evidence Act 1872, Police Act 1861, Special Powers Act 1974, plus any others you pick.
   Folder structure
   legal-qa/
   data/raw/ data/clean/ data/benchmark/
   src/ingest.py src/preprocess.py src/index.py src/retrieve.py
   src/procedure.py src/generate.py src/verify.py src/evaluate.py
   notebooks/ app.py results/ report/
   Stage 1: Data collection
   Inputs: BLAD (Kaggle), UKIL-DB-EN (Hugging Face), bdlaws for anything missing.
   Steps:
   Filter to your acts by name or act ID.
   For each act, keep every section's text in English and Bangla if both exist.
   Drop repealed sections (UKIL and BLAD have flags). Keep the amendment date if available.
   Output: sections.jsonl, one record per section.

Schema:

json
{"sec_uid":"pc1860_s302","act_id":11,"act_name":"Penal Code, 1860",
"section_no":"302","title":"Punishment for murder",
"text_en":"...","text_bn":"...","lang":"en|bn|mixed",
"repealed":false,"amended":"2026-..","footnotes":[]}

Link Bangla and English by sec_uid so a Bangla query can retrieve English text and vice versa.

Stage 2: Text preprocessing (core NLP content)

Describe and justify each step in the report.

Cleaning: strip footnote markers like [1], HTML leftovers, page numbers and repeated headers.
Unicode normalization: apply NFC, and unify Bangla digits to Arabic digits (০১২ → 012) for section numbers. Normalize variant characters (e.g. ড় vs ড + ়).
Language detection: simple Unicode-range check (Bangla block U+0980–09FF) to label each text segment bn, en or mixed.
Sentence and clause splitting: split Bangla on "।", English on ".", and keep sub-section markers like (1), (2), (a), (b).
Tokenization: compare whitespace tokenization with the embedding model's subword tokenizer. Show the token count difference between Bangla and English text.
Stopwords and stemming for BM25 only: use a small Bangla stopword list and optionally a light stemmer. Don't apply these to dense embeddings.
Chunking: one section equals one chunk. Split sections longer than about 400 tokens into overlapping chunks (50 tokens overlap) that keep sec_uid.
Legal dictionary: hand-build 40–80 terms (e.g. naraji, FIR/এজাহার, remand, bail/জামিন, cognizable/আমলযোগ্য, non-cognizable/অআমলযোগ্য) with English, Bangla and colloquial forms. Use it for query expansion.
Stage 3: Procedure table (novelty #2)

CrPC's Second Schedule lists offences with whether police may arrest without a warrant, whether it is bailable, and which court tries it.

Extract the table rows for Penal Code sections.
Parse into {penal_section, cognizable, bailable, trial_court, punishment}. If parsing is messy, hand-encode the 40–60 most common offences and state this in the report.
Store as procedure.json, keyed by section number.
Stage 4: Benchmark creation (novelty #1, your main contribution)
Size: target 150–300 questions.
Per question: {qid, question, lang_style, gold_sections[], gold_procedure, answer_type}.
Language styles (use all three):
formal English
formal Bangla
colloquial Bangla, e.g. "পুলিশ কি ওয়ারেন্ট ছাড়া ধরতে পারে?"
Question types: definition ("what is theft"), punishment, procedure ("can police arrest without warrant for X"), bail, evidence, scenario.
How to build it:
Write about 50 questions yourself from sections.
Use an LLM to draft more from random sections, then review every one manually.
Have a friend check 20–30 for gold-section correctness and compute simple agreement (Cohen's kappa) between you and the friend.
Splits: 30% dev (tune), 70% test (report). Never tune on the test set.
Stage 5: Indexing

Build three indexes so you can compare them:

BM25: rank_bm25, built on preprocessed tokens.
Dense, small: multilingual-e5-small through sentence-transformers. Use the prefixes query: and passage: that E5 expects.
Dense, Bangla-tuned (optional): kazalbrur/bangla-embed-e5-small.
Optional on Colab: bge-m3 as a stronger comparison (cached embeddings saved to file).

Store embeddings in FAISS (IndexFlatIP with normalized vectors). Embed once, save .npy files, and never re-embed.

Stage 6: Retrieval
Query processing: detect language, expand using the legal dictionary, then optionally translate Bangla queries to English with an LLM as an extra variant.
Hybrid fusion: reciprocal rank fusion:
python
def rrf(rank_lists, k=60):
scores = {}
for ranks in rank_lists:
for r, doc in enumerate(ranks):
scores[doc] = scores.get(doc, 0) + 1/(k + r + 1)
return sorted(scores, key=scores.get, reverse=True)
Exact-match boost: if the query contains "section 302" or "ধারা ৩০২", parse the number and boost that section.
Optional reranker: bge-reranker-v2-m3 on Colab only.
Output: top-5 sections with scores.
Stage 7: Procedure lookup

After retrieval, if the top section is a Penal Code section and the question is about arrest or bail, join with procedure.json and attach the structured row to the context as trusted facts.

Stage 8: Answer generation
LLM: a free-tier API for the final system. For an offline fallback, a 4-bit Qwen or Gemma at 1.7–4B via Ollama.
Prompt rules:
answer only from the provided sections and procedure row
cite section IDs in a fixed format, e.g. [Penal Code §302]
answer in the user's language
say "not found in the provided law" when the context doesn't answer
Settings: temperature 0–0.2, and cap the output length.
Stage 9: Citation verification (novelty #3)
Extract every citation from the answer with a regex.
Check that each cited sec_uid exists in the database and appeared in the retrieved context.
Optional: use NLI or an LLM judge to check the cited text supports the claim.
On failure, regenerate once with an error hint. If it still fails, abstain.
Stage 10: Evaluation (what the marks depend on)

Retrieval metrics (per language style): Recall@1/3/5, MRR, nDCG@5.

Systems to compare:

ID System
S1 BM25
S2 e5-small dense
S3 Bangla-tuned dense
S4 Hybrid (BM25 + dense)
S5 S4 + legal dictionary expansion

Generation metrics:

Citation validity rate (percent of cited sections that exist and were retrieved)
Faithfulness (LLM judge, with 50 samples checked by hand)
Procedure accuracy on arrest and bail questions
Abstention rate on out-of-scope questions (add 20 such questions)

Ablations: LLM-only vs RAG, RAG vs RAG + procedure lookup, RAG vs RAG + verifier.

Error analysis: categorize 30–50 failures, e.g. Bangla vocabulary mismatch, wrong act, section-number confusion, colloquial query failure, cross-reference missed. Include 3–4 worked examples.

Stage 11: Demo app

A small Streamlit app: query box, language selector, answer, retrieved sections, and the procedure card. Add a disclaimer: legal information, not legal advice. Keep it under about 100 lines.

Stage 12: Report structure (NLP course style)
Introduction and research questions
Related work (UKIL, Mina, LegalRAG, BLAD, Bar Council QA benchmark)
Data and preprocessing (with Bangla-specific issues)
Methodology (stages 5–9, with an architecture diagram)
Benchmark construction and agreement
Experiments and results tables
Error analysis
Limitations and ethics (hallucination risk, outdated law, not legal advice, license of data)
Conclusion and future work (fine-tuned embeddings, case law, Police Gazettes, speech input)
