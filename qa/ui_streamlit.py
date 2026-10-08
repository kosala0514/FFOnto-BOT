"""
Chat UI.   streamlit run qa/ui_streamlit.py      (FastAPI must be running on QA_API, default :8000)
"""
import os
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("QA_API", "http://localhost:8000")
st.set_page_config(page_title="FFOnto assistant", page_icon="🔥", layout="wide")


@st.cache_data(ttl=60)
def get_records():
    try:
        return requests.get(f"{API}/records", timeout=30).json()
    except Exception as e:
        st.sidebar.error(f"API not reachable: {e}")
        return []


with st.sidebar:
    st.header("Context")
    recs = get_records()
    options = ["(none)"] + [f"{r['type'].replace('Record', '')}: {r['label']}  [{r['event']}]" for r in recs]
    choice = st.selectbox("Image / audio record", options)
    rec = recs[options.index(choice) - 1] if choice != "(none)" else None
    if rec and rec["file"] and os.path.exists(rec["file"]):
        if rec["type"] == "ImageRecord":
            st.image(rec["file"], use_container_width=True)
        else:
            st.audio(rec["file"])
    show_evidence = st.checkbox("Show evidence (SPARQL + rows)", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("🔥 Forest-Watch")
st.caption("Answers come from the FFOnto knowledge graph in GraphDB (with charts of the result rows); "
           "general questions use ontology-grounded multimodal RAG over your documents' text, tables and figures.")
st.session_state.setdefault("messages", [])


def show_visuals(visuals):
    """Charts from the knowledge graph and figures from the documents, next to the text answer."""
    if not visuals:
        return
    cited = [v for v in visuals if v.get("kind") != "image" or v.get("cited")]
    related = [v for v in visuals if v.get("kind") == "image" and not v.get("cited")]

    def one(v):
        if v["kind"] == "vega":
            st.caption(f"📊 {v['title']} – from {v['source']}")
            st.vega_lite_chart(v["spec"], use_container_width=True)
        elif v["kind"] == "image":
            src = v["path"] if os.path.exists(v.get("path", "")) else API + v["url"]
            st.image(src, caption=f"{v['title']}  ·  {v['source']}", use_container_width=True)
            if v.get("reading"):
                st.caption(f"Visual reading: {v['reading']}")
        elif v["kind"] == "note":
            st.caption(f"({v['title']}: {v.get('text', '')})")

    cols = st.columns(min(2, len(cited))) if len(cited) > 1 else [st.container()]
    for i, v in enumerate(cited):
        with cols[i % len(cols)]:
            one(v)
    if related:
        with st.expander(f"Related figures not used in the answer ({len(related)})"):
            for v in related:
                one(v)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("evidence"):
            show_visuals(m["evidence"].get("visuals"))
        if m.get("evidence") and show_evidence:
            ev = m["evidence"]
            with st.expander(f"Evidence – route: {ev.get('route')}, mode: {ev.get('mode')}, CQ: {ev.get('cq')}, {ev.get('seconds')} s"):
                if ev.get("filters"):
                    st.write("Filters:", ev["filters"])
                if ev.get("sparql"):
                    st.code(ev["sparql"], language="sparql")
                if ev.get("rows"):
                    st.dataframe(pd.DataFrame(ev["rows"]))
                if ev.get("sources"):
                    for s in ev["sources"]:
                        tag = f" [{s['ref']}]{' ✓ cited' if s.get('cited') else ''}" if s.get("ref") else ""
                    st.markdown(f"- **{s['type']}**{tag} {s.get('source', '')}: {s['text']}")
                if ev.get("decision", {}).get("examples"):
                    st.write("Similar labelled questions:", ev["decision"]["examples"])
                if ev.get("error"):
                    st.error(ev["error"])

if prompt := st.chat_input("Ask about the selected image/audio, an event, weather, or wildfires in general…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                res = requests.post(f"{API}/chat", json={"message": prompt, "record": rec["iri"] if rec else None,
                                                          "record_label": rec["label"] if rec else None},
                                    timeout=600).json()
            except Exception as e:
                res = {"answer": f"API error: {e}"}
        st.markdown(res.get("answer", ""))
    st.session_state.messages.append({"role": "assistant", "content": res.get("answer", ""), "evidence": res})
    st.rerun()
