"""
The orchestrator.

    question ─► router (LLM + similar labelled questions)
               ├─ kg       ─► CQ template (tested SPARQL)  ─► GraphDB ─► answer from rows only
               │              └─ no CQ fits ─► text-to-SPARQL (validated, 1 repair) ─► GraphDB ─► answer
               ├─ rag      ─► ontology facts + document passages + document figures (multimodal RAG)
               │              └─ VLM re-reads the top figures for this question ─► answer citing [1] [F1] [O]
               └─ chitchat ─► short help text
Every response carries its evidence (route, CQ, SPARQL, rows, sources) so answers are traceable,
and `visuals`: charts drawn from the KG rows, or the document figures the answer cites.
"""
import time
import re
from . import config, llm, kg, cqs, prompts, rag, visuals
from .examples import ExampleIndex

_index = None
_schema = None


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def example_index():
    global _index
    if _index is None:
        _index = ExampleIndex.load() or False
    return _index or None


# ------------------------------------------------------------------ routing
def route(question, record_desc="none", exclude_self=False):
    idx = example_index()
    nearest = idx.nearest(question, exclude_text=question if exclude_self else None) if idx else []
    ex_txt = "\n".join(f'- "{e["question"]}" -> {e["cq"] if e["route"] == "kg" else "rag"}' for e in nearest) or "(none)"
    user = fill(prompts.ROUTER_USER, catalog=cqs.catalog_text(), examples=ex_txt,
                record=record_desc, question=question)
    decision = llm.chat_json(prompts.ROUTER_SYSTEM, user, prompts.router_schema(cqs.REGISTRY.keys()))
    decision["examples"] = [{"question": e["question"], "label": e["cq"], "score": round(e["score"], 3)} for e in nearest]
    return decision


# ------------------------------------------------------------------ KG answering
def _table(cols, rows, limit):
    show = rows[:limit]
    if not show:
        return "(no rows)"
    lines = [" | ".join(cols)] + [" | ".join(str(r.get(c, ""))[:80] for c in cols) for r in show]
    return "\n".join(lines)


def answer_from_rows(question, title, filters, cols, rows):
    n = len(rows)
    user = fill(prompts.ANSWER_KG_USER, question=question, title=title, filters=filters or "none", n=n,
                truncated=f", first {config.MAX_ROWS_TO_LLM} shown" if n > config.MAX_ROWS_TO_LLM else "",
                table=_table(cols, rows, config.MAX_ROWS_TO_LLM))
    return llm.chat_text(prompts.ANSWER_KG_SYSTEM, user)


def schema_summary():
    """Compact description of the core ontology for the text-to-SPARQL prompt."""
    global _schema
    if _schema:
        return _schema
    from rdflib import Graph, RDF, RDFS, OWL, URIRef
    g = Graph().parse(config.ONTOLOGY_DIR / "ffonto-core.ttl", format="turtle")
    nm = g.namespace_manager

    def q(t):
        if t is None:
            return "?"
        if isinstance(t, URIRef):
            return t.n3(nm)
        members = list(g.items(g.value(t, OWL.unionOf))) if g.value(t, OWL.unionOf) else []
        return " or ".join(m.n3(nm) for m in members) or "?"

    lines = ["Classes:"]
    for c in sorted(g.subjects(RDF.type, OWL.Class), key=str):
        if isinstance(c, URIRef) and str(c).startswith(config.PREFIXES["ffo"]):
            sup = ", ".join(q(s) for s in g.objects(c, RDFS.subClassOf) if isinstance(s, URIRef))
            com = str(g.value(c, RDFS.comment) or "")[:110]
            lines.append(f"  {q(c)}{' subClassOf ' + sup if sup else ''} – {com}")
    lines.append("Properties (domain -> range):")
    for t in (OWL.ObjectProperty, OWL.DatatypeProperty):
        for p in sorted(g.subjects(RDF.type, t), key=str):
            if str(p).startswith(config.PREFIXES["ffo"]):
                lines.append(f"  {q(p)} ({q(g.value(p, RDFS.domain))} -> {q(g.value(p, RDFS.range))})")
    lines.append("Also used: sosa:observedProperty, sosa:hasFeatureOfInterest, prov:wasAttributedTo, "
                 "prov:wasDerivedFrom, skos:prefLabel, skos:broader, rdfs:label.")
    _schema = "\n".join(lines)
    return _schema


def text_to_sparql(question, record, hint_cqs):
    examples = "\n\n".join(f"# {c} – {cqs.REGISTRY[c].title}\n{cqs.REGISTRY[c].query.split('PREFIX dcterms: <http://purl.org/dc/terms/>')[-1].strip()}"
                           for c in hint_cqs if c in cqs.REGISTRY)[:6000]
    system = fill(prompts.TEXT2SPARQL_SYSTEM, schema=schema_summary(), examples=examples or "(none)")
    error, last = "", None
    for attempt in range(2):
        out = llm.chat_json(system, fill(prompts.TEXT2SPARQL_USER, record=record or "none",
                                         question=question, error=error),
                            {"type": "object", "properties": {"sparql": {"type": "string"}}, "required": ["sparql"]},
                            model=config.SPARQL_MODEL)
        try:
            q = kg.make_safe(out["sparql"])
            cols, rows = kg.select(q)
            return q, cols, rows
        except kg.KGError as e:
            last = e
            error = f"Your previous query failed: {e}\nPrevious query:\n{out.get('sparql', '')}\nFix it."
    raise last


# ------------------------------------------------------------------ RAG
def visual_reread(question, figures, n=None):
    """Let the vision model look at the top figures again, this time with the question in mind.
    The stored description was written without knowing the question and may miss the value that is asked for."""
    for f in figures[: (config.N_REREAD if n is None else n)]:
        try:
            txt = llm.chat_vision(prompts.VISUAL_REREAD_SYSTEM,
                                  fill(prompts.VISUAL_REREAD_USER, question=question, source=f["source"],
                                       caption=f.get("caption") or "(none)"), [f["image"]])
        except (llm.LLMError, OSError) as e:
            txt = f"(not available: {e})"
        f["reading"] = "" if txt.strip().upper().startswith("NOT RELEVANT") else txt.strip()
    return figures


def rag_answer(question, use_figures=True):
    facts = rag.ontology_facts(question)
    passages, figures = rag.retrieve_mm(question, k_fig=None if use_figures else 0)
    if figures and config.VISUAL_REREAD:
        visual_reread(question, figures)
    fig_txt = "\n\n".join(
        f"[F{i + 1}] ({f['source']}) {f['text'][:1800]}" + (f"\nVisual reading for this question: {f['reading']}"
                                                         if f.get("reading") else "")
        for i, f in enumerate(figures)) or "(none)"
    user = fill(prompts.ANSWER_RAG_USER, question=question,
                facts="\n".join(f"- {f}" for f in facts) or "(none)",
                passages="\n\n".join(f"[{i + 1}] ({p['source']}) {p['text']}" for i, p in enumerate(passages)) or "(none)",
                figures=fig_txt)
    answer = llm.chat_text(prompts.ANSWER_RAG_SYSTEM, user)
    cited = {int(n) for n in re.findall(r"\[F(\d+)\]", answer)}
    for i, f in enumerate(figures):
        f["ref"] = f"F{i + 1}"
        f["cited"] = (i + 1) in cited
    return answer, facts, passages, figures


def figure_visuals(figures):
    """Figures the answer cites come first; uncited ones are returned as 'related' (the UI collapses them)."""
    out = []
    for f in sorted(figures, key=lambda f: (not f.get("cited"), -f["score"])):
        out.append({"kind": "image", "ref": f["ref"], "title": f"[{f['ref']}] {f.get('caption') or f['source']}"[:160],
                    "path": f["image"], "url": f"/figures/{f['file']}", "source": f["source"],
                    "figure_type": f.get("figure_type"), "cited": f.get("cited", False),
                    "score": round(f["score"], 3), "reading": f.get("reading", "")})
    return out


# ------------------------------------------------------------------ main entry
def answer(question, record=None, record_desc=None, force_cq=None, force_text2sparql=None):
    t0 = time.time()
    out = {"question": question, "record": record, "route": None, "cq": None, "mode": None,
           "sparql": None, "columns": [], "rows": [], "filters": {}, "answer": "", "sources": [], "visuals": [], "error": None}
    try:
        if force_cq:
            decision = {"route": "kg", "cq": force_cq, "uses_selected_record": True, "event": "", "site": "", "date": "",
                        "reason": "forced", "examples": []}
        else:
            decision = route(question, record_desc or (kg.short(record) if record else "none"))
        out["decision"] = decision
        out["route"] = decision["route"]

        if decision["route"] == "chitchat":
            out["mode"] = "chitchat"
            out["answer"] = llm.chat_text(prompts.CHITCHAT_SYSTEM, question)

        elif decision["route"] == "rag":
            out["mode"] = "rag"
            out["answer"], facts, passages, figures = rag_answer(question)
            out["sources"] = [{"type": "ontology", "text": f} for f in facts] + \
                             [{"type": "document" if p["type"] == "text" else p["type"], "source": p["source"],
                               "text": p["text"][:300]} for p in passages] + \
                             [{"type": "figure", "source": f["source"], "ref": f["ref"], "cited": f["cited"],
                               "text": (f["text"][:300] + (" | reading: " + f["reading"][:200] if f.get("reading") else ""))}
                              for f in figures]
            out["visuals"] = figure_visuals(figures)

        else:
            baseline = config.FORCE_TEXT2SPARQL if force_text2sparql is None else force_text2sparql
            cq_id = "NONE" if baseline else decision["cq"]
            out["router_cq"] = decision["cq"]
            use_rec = record if decision.get("uses_selected_record") or force_cq else None
            if cq_id != "NONE":
                cq = cqs.REGISTRY[cq_id]
                if "record" in cq.focus and not use_rec and not decision.get("event") and decision.get("uses_selected_record"):
                    out["mode"] = "cq"; out["cq"] = cq_id
                    out["answer"] = "Please select an image or audio record first – this question is about a specific record."
                    return out
                res = cqs.run_cq(cq_id, record=use_rec, event=decision.get("event") or None,
                                 site=decision.get("site") or None, date=decision.get("date") or None)
                out.update(mode="cq", cq=cq_id, sparql=res["query"], columns=res["columns"],
                           rows=res["rows"][:200], filters=res["filters"])
                out["answer"] = answer_from_rows(question, res["title"], res["filters"], res["columns"], res["rows"])
                out["visuals"] = visuals.chart_for(cq_id, res["columns"], res["rows"], res["title"])
            elif config.ENABLE_TEXT2SPARQL or baseline:
                hints = [e["label"] for e in decision.get("examples", []) if e["label"] in cqs.REGISTRY][:3]
                q, cols, rows = text_to_sparql(question, use_rec or record, hints)
                out.update(mode="text2sparql", sparql=q, columns=cols, rows=rows[:200])
                out["answer"] = answer_from_rows(question, "generated SPARQL (no matching CQ)", {}, cols, rows)
                out["visuals"] = visuals.chart_for(None, cols, rows, "Query result")
            else:
                out["mode"] = "none"
                out["answer"] = "This question is not covered by the knowledge graph's competency questions."
    except (llm.LLMError, kg.KGError, KeyError) as e:
        out["error"] = str(e)
        out["answer"] = f"Sorry – I could not answer that ({e})."
    finally:
        out["seconds"] = round(time.time() - t0, 2)
    return out
