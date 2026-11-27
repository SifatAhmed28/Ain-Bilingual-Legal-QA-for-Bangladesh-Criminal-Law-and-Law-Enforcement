"""Build persistent BM25 and optional multilingual dense retrieval indexes."""
from __future__ import annotations
import json, pickle, re
from pathlib import Path
import numpy as np
from . import config as C

OUT = C.ROOT / "indexes"
def read_jsonl(path):
    with open(path, encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def tokens(s): return re.findall(r"[\w\u0980-\u09ff]+", s.casefold())
def build(dense=True):
    docs=read_jsonl(C.DATA_CLEAN/"chunks.jsonl")
    OUT.mkdir(exist_ok=True)
    with (OUT/"chunks.jsonl").open("w",encoding="utf-8") as f:
        for d in docs: f.write(json.dumps(d,ensure_ascii=False)+"\n")
    corpus=[tokens(d.get("text_bm25") or d["text"]) for d in docs]
    # rank_bm25 is optional; storing tokens keeps the index portable.
    with (OUT/"bm25_tokens.pkl").open("wb") as f: pickle.dump(corpus,f)
    meta={"documents":len(docs),"bm25":"token corpus; BM25Okapi at query time","dense_model":C.EMBED_MODEL if dense else None}
    if dense:
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(C.EMBED_MODEL)
        vectors=model.encode(["passage: "+d["text"] for d in docs],normalize_embeddings=True,show_progress_bar=True)
        np.save(OUT/"dense_embeddings.npy",np.asarray(vectors,dtype="float32"))
        try:
            import faiss
            idx=faiss.IndexFlatIP(vectors.shape[1]); idx.add(np.asarray(vectors,dtype="float32")); faiss.write_index(idx,str(OUT/"dense.faiss"))
            meta["faiss"]=True
        except ImportError: meta["faiss"]=False
    (OUT/"manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--bm25-only",action="store_true"); a=p.parse_args(); print(build(not a.bm25_only))
