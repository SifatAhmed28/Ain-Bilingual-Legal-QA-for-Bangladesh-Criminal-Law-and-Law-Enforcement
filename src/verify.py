"""Deterministic citation checks against known and retrieved section ids."""
import re
def citations(answer):
    return re.findall(r"\[([^\]]+?)\s*§\s*([\w.-]+)\]",answer,re.I)
def verify(answer, known_ids, retrieved_ids):
    found=citations(answer); valid=[]; errors=[]
    for act,sec in found:
        key=(act.strip(),sec.strip())
        if key not in known_ids: errors.append({"citation":key,"error":"unknown_section"})
        elif key not in retrieved_ids: errors.append({"citation":key,"error":"not_retrieved"})
        else: valid.append(key)
    return {"valid":not errors,"citations":found,"errors":errors}
