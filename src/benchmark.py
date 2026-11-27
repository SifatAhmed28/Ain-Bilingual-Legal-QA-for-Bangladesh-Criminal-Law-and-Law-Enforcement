"""Create a reproducible benchmark seed; generated items require human review."""
import json
from . import config as C
def build():
    rows=[json.loads(x) for x in (C.DATA_CLEAN/"sections.jsonl").open(encoding="utf-8")]
    candidates=[r for r in rows if r["act_id"] in (11,75) and r.get("section_no")!="preamble" and r.get("title") and "omitted" not in r["title"].lower() and len(r["title"])>8][:50]
    output=[]
    for r in candidates:
        title=r["title"].strip(" []\"'“”")
        for style,q in (("formal_english",f"Which provision covers {title.lower()}?"),("formal_bangla",f"{title} সম্পর্কে কোন ধারায় বলা হয়েছে?"),("colloquial_bangla",f"{title} কোন ধারায় আছে?")):
            n=len(output); output.append({"qid":f"seed_{n+1:04d}","question":q,"lang_style":style,"gold_sections":[r["sec_uid"]],"gold_procedure":None,"answer_type":"definition","split":"dev" if n%10<3 else "test","review_status":"needs_manual_review","source_sec_uid":r["sec_uid"]})
    C.DATA_BENCHMARK.mkdir(parents=True,exist_ok=True); p=C.DATA_BENCHMARK/"questions.jsonl"
    p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in output),encoding="utf-8"); return len(output)
if __name__=="__main__": print(f"Wrote {build()} benchmark seed questions; all marked needs_manual_review.")
