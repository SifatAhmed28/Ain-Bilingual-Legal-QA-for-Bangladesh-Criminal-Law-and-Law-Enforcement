"""Streamlit search interface for the Bangladesh legal corpus."""
from html import escape

import streamlit as st

from src.retrieve import Retriever


st.set_page_config(
    page_title="Ain — Bangladesh Legal Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#17251f; --muted:#68766e; --green:#176b50; --green-dark:#10543f; --mint:#eaf4ee; --line:#e3eae5; --paper:#f7f9f7; }
      .stApp { background:var(--paper); color:var(--ink); }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { max-width:1080px; padding:2.1rem 2rem 4rem; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      h1,h2,h3 { color:var(--ink); font-family:'Manrope','DM Sans',sans-serif; }
      .topbar { display:flex; align-items:center; gap:.75rem; margin-bottom:3.3rem; }
      .logo { width:42px;height:42px;border-radius:14px;background:var(--green);color:white;display:grid;place-items:center;font-size:21px;box-shadow:0 7px 18px #176b5025; }
      .wordmark { font-family:'Manrope',sans-serif;font-weight:800;font-size:1.06rem;letter-spacing:-.04em; }
      .wordmark small { display:block;color:var(--muted);font-family:'DM Sans',sans-serif;font-weight:400;font-size:.72rem;letter-spacing:0;margin-top:1px; }
      .eyebrow { color:var(--green);font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.65rem; }
      .hero { font-family:'Manrope',sans-serif;font-size:clamp(2.25rem,5vw,3.65rem);font-weight:700;line-height:1.08;letter-spacing:-.065em;margin:0; }
      .intro { max-width:650px;color:var(--muted);font-size:1rem;line-height:1.7;margin:.9rem 0 1.8rem; }
      div[data-testid="stForm"] { background:white;border:1px solid var(--line);border-radius:20px;padding:1rem 1.1rem .65rem;box-shadow:0 14px 44px #1a30220b; }
      div[data-testid="stTextInput"] input { border:0!important;box-shadow:none!important;font-size:1.02rem;padding:.65rem .25rem; }
      div[data-testid="stTextInput"] input:focus { outline:none!important; }
      div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button { background:var(--green);border:0;border-radius:11px;color:white;font-weight:600;padding:.65rem 1rem; }
      div.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button:hover { background:var(--green-dark);border:0;color:white; }
      div[data-testid="stForm"] [data-testid="column"] { align-self:center; }
      .suggest-label { color:var(--muted);font-size:.78rem;margin:.85rem 0 .4rem; }
      div[data-testid="stButton"] button[kind="secondary"] { border:1px solid var(--line);border-radius:999px;background:white;color:#34463d;font-size:.78rem;min-height:2.25rem; }
      div[data-testid="stButton"] button[kind="secondary"]:hover { border-color:#9bb9a8;color:var(--green-dark); }
      .result-head { display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:2.8rem 0 .8rem; }
      .result-head h2 { font-size:1.25rem;letter-spacing:-.035em;margin:0; }
      .result-meta { color:var(--muted);font-size:.8rem; }
      .best-card { background:white;border:1px solid var(--line);border-radius:18px;padding:1.2rem 1.35rem;box-shadow:0 10px 32px #1a302208; }
      .best-label { color:var(--green);font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.65rem; }
      .best-title { font-family:'Manrope',sans-serif;font-size:1.08rem;font-weight:700;letter-spacing:-.025em;margin-bottom:.7rem; }
      .best-text { color:#34443b;line-height:1.8;white-space:pre-wrap;font-size:.94rem; }
      .query-chip { display:inline-block;background:var(--mint);color:var(--green-dark);border-radius:999px;padding:.38rem .72rem;font-size:.76rem;font-weight:600;margin:.15rem 0 .4rem; }
      .notice { border:1px solid var(--line);border-radius:14px;background:#fff;padding:.85rem 1rem;color:var(--muted);font-size:.78rem;line-height:1.6;margin-top:1.6rem; }
      div[data-testid="stExpander"] { background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-bottom:.55rem; }
      div[data-testid="stExpander"] details summary p { font-weight:600;font-size:.9rem; }
      .empty-card { text-align:center;background:white;border:1px dashed #cad7ce;border-radius:18px;padding:2rem 1rem;color:var(--muted);margin-top:2rem; }
      .footer { border-top:1px solid var(--line);margin-top:3rem;padding-top:1rem;color:#7a867f;font-size:.72rem; }
      @media(max-width:700px) { .block-container{padding:1.4rem 1rem 3rem}.topbar{margin-bottom:2.4rem}.hero{font-size:2.35rem}.result-head{align-items:flex-start;flex-direction:column;gap:.25rem} }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the legal search index…")
def load_retriever():
    # Keep the lightweight, local demo independent of dense-model downloads.
    return Retriever(dense=False)


def run_search(query: str) -> None:
    """Search once and persist the result while Streamlit reruns the page."""
    clean_query = query.strip()
    if not clean_query:
        st.session_state.pop("search_result", None)
        st.session_state["search_error"] = "Enter a question or section number to search."
        return
    st.session_state.pop("search_error", None)
    try:
        with st.spinner("Searching the law collection…"):
            hits = load_retriever().search(clean_query, k=5, system="hybrid_expanded")
        st.session_state["search_result"] = {"question": clean_query, "hits": hits}
    except Exception as exc:
        st.session_state.pop("search_result", None)
        st.session_state["search_error"] = f"Search could not be completed: {exc}"


def choose_example(sample: str) -> None:
    """Fill the search box and immediately show results for a sample query."""
    st.session_state["question_input"] = sample
    run_search(sample)


st.markdown(
    '<div class="topbar"><div class="logo">⚖</div><div class="wordmark">Ain<small>Bangladesh legal search</small></div></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">Bangladesh · criminal law & procedure</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero">Find the law behind<br>your question.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="intro">Search criminal law, procedure, evidence and policing statutes in English or Bangla. Review the original provisions behind every result.</p>',
    unsafe_allow_html=True,
)

with st.form("legal_search", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="Ask a question or enter a section number…  /  প্রশ্ন লিখুন",
        label_visibility="collapsed",
    )
    search_col, hint_col = st.columns([1.25, 4.75])
    with search_col:
        submitted = st.form_submit_button("Search laws  →", type="primary", use_container_width=True)
    with hint_col:
        st.caption("Search by topic, offence, legal term, or section number — for example, section 302.")

if submitted:
    run_search(question)

st.markdown('<div class="suggest-label">Try an example</div>', unsafe_allow_html=True)
suggestions = [
    "Can police arrest without a warrant?",
    "How does bail work?",
    "What does section 302 cover?",
]
suggestion_cols = st.columns(len(suggestions))
for col, sample in zip(suggestion_cols, suggestions):
    with col:
        st.button(
            sample,
            key=f"suggest_{sample}",
            use_container_width=True,
            on_click=choose_example,
            args=(sample,),
        )

if st.session_state.get("search_error"):
    st.warning(st.session_state["search_error"])

result = st.session_state.get("search_result")
if result:
    hits = result["hits"]
    safe_query = escape(result["question"])
    if not hits:
        st.markdown(
            '<div class="empty-card"><strong>No matching provisions found</strong><br>Try a shorter phrase, a legal term, or a section number.</div>',
            unsafe_allow_html=True,
        )
    else:
        best = hits[0]
        act = str(best.get("act_name", "Act"))
        section = str(best.get("section_no", ""))
        title = str(best.get("title", ""))
        citation = escape(f"{act} §{section}")
        heading = escape(title) if title else citation
        excerpt = escape(str(best.get("text", "")).strip()[:1600])
        lang = str(best.get("lang", "source text")).upper()

        st.markdown(
            f'<div class="result-head"><h2>Search results</h2><span class="result-meta">{len(hits)} relevant provisions</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<span class="query-chip">{safe_query}</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="best-card"><div class="best-label">Best matching provision · {escape(lang)} source</div>'
            f'<div class="best-title">{heading}</div><div class="result-meta">{citation}</div>'
            f'<div class="best-text" style="margin-top:.8rem">{excerpt}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="result-head"><h2>Other matching provisions</h2><span class="result-meta">Open a result to read its full indexed passage</span></div>', unsafe_allow_html=True)
        for rank, doc in enumerate(hits[1:], start=2):
            act_name = str(doc.get("act_name", "Act"))
            sec_no = str(doc.get("section_no", ""))
            doc_title = str(doc.get("title", ""))
            label = f"{rank:02d}  ·  {act_name} §{sec_no}" + (f"  —  {doc_title}" if doc_title else "")
            with st.expander(label):
                st.caption(f"{str(doc.get('lang', 'source text')).upper()} source · {act_name} §{sec_no}")
                st.write(doc.get("text", ""))

        with st.expander("Read the full top passage"):
            st.caption(f"{lang} source · {act} §{section}")
            st.write(best.get("text", ""))

st.markdown(
    '<div class="notice"><strong>Research aid only.</strong> Search results are original legal text, not a legal determination or legal advice. Laws may be amended; confirm the current wording in an official source.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="footer">Ain · Bangladesh Legal Search · Local BM25 retrieval</div>', unsafe_allow_html=True)
