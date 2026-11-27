"""BM25, multilingual E5 dense, RRF hybrid and exact-section retrieval."""
import json,pickle,re
from pathlib import Path
import numpy as np
from .index import tokens,OUT
from . import config as C
from .lexicon import load_lexicon,expand_query,expanded_query_text
def rrf(rank_lists,k=60):
    scores={}
    for ranks in rank_lists:
        for rank,doc in enumerate(ranks): scores[doc]=scores.get(doc,0)+1/(k+rank+1)
    return sorted(scores,key=scores.get,reverse=True)
class Retriever:
    def __init__(self, dense=True):
        self.docs=[json.loads(x) for x in (OUT/"chunks.jsonl").open(encoding="utf-8")]
        self.corpus=pickle.load((OUT/"bm25_tokens.pkl").open("rb"))
        try:
            from rank_bm25 import BM25Okapi
            self.bm=BM25Okapi(self.corpus)
        except ImportError: self.bm=None
        self.dense=dense and (OUT/"dense_embeddings.npy").exists()
        if self.dense:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(C.EMBED_MODEL); self.vectors=np.load(OUT/"dense_embeddings.npy")
        self.lex=load_lexicon(C.DATA_LEXICON/"legal_terms.json")
    def search(self,q,k=5,system="hybrid",expand=True):
        query=q
        # Queries such as "someone thread me" commonly contain a typo for
        # "threaten". Map threat language to the Penal Code provisions so
        # generic words ("what", "can", "do") cannot dominate BM25.
        threat_query = bool(re.search(
            r"\b(threat|threats|threaten|threatens|threatened|threatening|thread|threads|intimidat\w*)\b|হুমকি|ভয় দেখ|ভয় দেখ|প্রাণনাশের হুমকি",
            q, re.I,
        ))
        if threat_query:
            query += " criminal intimidation threat injury alarm section 503 506"
            if re.search(r"anonymous|unknown|নামহীন|বেনামি|গোপনে", q, re.I):
                query += " anonymous communication section 507"
        if expand:
            try: query = expanded_query_text(q,expand_query(q,self.lex))
            except Exception: pass
        qt=tokens(query)
        if self.bm: bs=self.bm.get_scores(qt)
        else:
            # BM25 with Robertson IDF and length normalisation.
            n=len(self.corpus); avg=sum(map(len,self.corpus))/max(n,1); df={t:sum(t in set(d) for d in self.corpus) for t in set(qt)}; bs=[]
            for d in self.corpus:
                tf={t:d.count(t) for t in set(qt)}; score=sum(np.log(1+(n-df[t]+.5)/(df[t]+.5))*tf[t]*2.2/(tf[t]+1.2*(.25+.75*len(d)/max(avg,1))) for t in tf); bs.append(score)
            bs=np.asarray(bs)
        br=list(np.argsort(bs)[::-1]); rankings=[br]
        if self.dense and system!="bm25":
            v=self.model.encode(["query: "+q],normalize_embeddings=True)[0]; ds=self.vectors@v; dr=list(np.argsort(ds)[::-1]); rankings.append(dr)
        if system=="bm25": order=br
        elif system=="dense" and len(rankings)>1: order=rankings[-1]
        elif system in ("hybrid","hybrid_expanded"): order=rrf(rankings)
        else: order=br
        m=re.search(r"(?:section|sec(?:tion)?\.?|ধারা)\s*(\d+[a-z]?)",q,re.I)
        if m:
            number=m.group(1).upper()
            exact=[i for i,d in enumerate(self.docs) if d.get("section_no","").upper()==number]
            if exact: order=exact+ [i for i in order if i not in exact]
        if threat_query:
            priority_sections=["503","506"]
            if re.search(r"anonymous|unknown|নামহীন|বেনামি|গোপনে", q, re.I):
                priority_sections.insert(1,"507")
            priority=[i for section in priority_sections for i,d in enumerate(self.docs)
                      if d.get("act_id")==11 and d.get("section_no","").upper()==section]
            # For a clear threat query, suppress lexical noise such as
            # "house-breaking" that happens to share generic question words.
            order=priority
        # Return one best chunk per section: long laws otherwise occupy
        # several result slots and hide other useful provisions.
        results=[]; seen_sections=set()
        for i in order:
            doc=self.docs[i]
            key=doc.get("sec_uid",str(i))
            if key in seen_sections: continue
            seen_sections.add(key)
            results.append({**doc,"score":float(bs[i])})
            if len(results)>=k: break
        return results
