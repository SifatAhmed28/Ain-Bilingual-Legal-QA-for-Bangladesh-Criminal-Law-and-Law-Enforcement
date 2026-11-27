"""Prompt construction and extractive fallback; no API key or model required."""
def make_prompt(question, contexts, procedure=None):
    facts="\n\n".join(f"[{x['act_name']} §{x['section_no']}] {x['text']}" for x in contexts)
    if procedure: facts += "\n\nTrusted procedure row: "+str(procedure)
    return ("Answer only from these legal excerpts. Reply in the question's language. "
            "Cite each legal claim as [Act name §section]. If the excerpts do not answer, "
            "say 'not found in the provided law'. Do not infer legal advice.\n\n"
            f"Question: {question}\n\nExcerpts:\n{facts}\n\nAnswer:")
def extractive_answer(question, contexts):
    if not contexts: return "Not found in the provided law."
    d=contexts[0]; txt=d.get("text","").strip()
    return f"{txt[:1000]}\n[{d.get('act_name','Unknown Act')} §{d.get('section_no','?')}]" if txt else "Not found in the provided law."
