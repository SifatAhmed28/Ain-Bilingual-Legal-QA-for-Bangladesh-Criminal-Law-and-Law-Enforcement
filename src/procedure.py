"""CrPC Second Schedule procedure lookup. Unknown fields stay null."""
import json,re
from pathlib import Path
from . import config as C

def build():
    sections=[json.loads(x) for x in (C.DATA_CLEAN/"sections.jsonl").open(encoding="utf-8")]
    # Schedule extraction is source-layout dependent. Preserve the actual schedule
    # source as provenance and avoid inferring unknown classifications.
    schedule=next((s for s in sections if s["act_id"]==75 and re.search(r"second schedule|second schedule",str(s.get("title","")),re.I)),None)
    rows={}
    if schedule:
        text=schedule.get("text_en") or schedule.get("text_bn") or ""
        for m in re.finditer(r"(?m)^\s*(\d+[A-Z]?)\s*[.|:]\s*(.+)$",text):
            sec=m.group(1); row=m.group(2)
            rows[sec]={"penal_section":sec,"cognizable":None,"bailable":None,"trial_court":None,"punishment":None,"source":"CrPC Second Schedule","source_text":row,"verified":False}
    out=C.DATA_CLEAN/"procedure.json"; out.write_text(json.dumps({"source_act_id":75,"source":"CrPC Second Schedule (bdlaws corpus)","rows":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    return len(rows)
if __name__=="__main__": print(f"Extracted {build()} schedule rows; unparsed facts intentionally remain unknown.")
