"""
Ontology-grounded RAG for general-knowledge questions.

Three evidence sources are combined:
  1. ONTOLOGY FACTS – definitions and relations of the FFOnto concepts mentioned in the question
     (labels/keywords from ffonto-vocab.ttl, rdfs:comment from ffonto-core.ttl, plus links such as
     hazard -> mitigatedBy -> action, ecosystem -> typicalFireImpact -> impact).
  2. DOCUMENT PASSAGES – text chunks and tables of your own documents in qa/rag_docs/ (txt, md, pdf).
  3. DOCUMENT FIGURES – charts, maps, diagrams and photos cut out of the PDFs (or images placed in
     qa/rag_docs), described by a local vision-language model and tagged with ontology concepts
     (see qa/mm_pdf.py). Retrieved with their own quota; the PNG is returned to the UI.
The answer prompt tells the model to use only these sources.
"""
import json
import re
import numpy as np
from . import config, llm

DOC_FILE = config.INDEX_DIR / "rag_chunks.jsonl"
DOC_VEC = config.INDEX_DIR / "rag_chunks.npy"


# ------------------------------------------------------------------ documents
def _read(path):
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"skip {path.name}: pip install pypdf to index PDFs")
            return ""
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text, size=900, overlap=150):
    text = re.sub(r"[ \t]+", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            while len(p) > size:
                chunks.append(p[:size]); p = p[size - overlap:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def _text_items(path):
    """Text (and, with PyMuPDF, tables + figures) of one document."""
    from . import mm_pdf
    items, figures = [], []
    suf = path.suffix.lower()
    if suf == ".pdf" and mm_pdf.available():
        pages, tables, figures = mm_pdf.extract_pdf(path)
        for pg in pages:
            for i, c in enumerate(chunk(pg["text"])):
                items.append({"type": "text", "text": c, "source": f"{path.name} p.{pg['page']}#{i}",
                              "doc": path.name, "page": pg["page"]})
        for t in tables:
            items.append({"type": "table", "text": (f"Table from {t['doc']} (page {t['page']}). "
                                                     f"{t['caption']}\n{t['markdown']}")[:3000],
                          "source": f"{t['doc']} p.{t['page']} (table)", "doc": t["doc"], "page": t["page"]})
    elif suf in mm_pdf.IMAGE_EXT:
        figures = [mm_pdf.standalone_image(path)]
    else:
        if suf == ".pdf":
            print(f"note: {path.name}: text only (pip install pymupdf to also index its figures and tables)")
        for i, c in enumerate(chunk(_read(path))):
            items.append({"type": "text", "text": c, "source": f"{path.name}#{i}", "doc": path.name, "page": None})
    return items, figures


def build_doc_index(extra_chunks=None, figures=True, vision_model=None):
    """Index qa/rag_docs/*: text chunks, tables and (if figures=True) figures described by the VLM.
    extra_chunks: optional list of {'text','source'} (e.g. curated Q&A)."""
    from . import mm_pdf
    items, figs = [], []
    exts = {".txt", ".md", ".pdf"} | (mm_pdf.IMAGE_EXT if figures else set())
    for p in sorted(config.RAG_DOCS_DIR.glob("**/*")):
        if p.is_file() and p.suffix.lower() in exts and p.name.lower() != "readme.md":
            it, fg = _text_items(p)
            items += it
            figs += fg if figures else []
            print(f"  {p.name}: {len(it)} text/table chunks, {len(fg) if figures else 0} figures")
    if figs:
        print(f"describing {len(figs)} figures with {vision_model or config.VISION_MODEL} (cached after the first run) …")
        mm_pdf.describe_figures(figs, model=vision_model)
        for f in figs:
            v = f.get("vlm", {})
            f["concepts"] = [{"iri": str(i), "label": l} for i, l in
                             match_concepts(" ".join([f.get("pdf_caption", ""), v.get("title", ""),
                                                      v.get("description", ""), " ".join(v.get("key_findings", []))]),
                                            max_concepts=8)]
            items.append({"type": "figure", "text": mm_pdf.figure_text(f), "source": f"{f['doc']} p.{f['page']} (figure)",
                          "doc": f["doc"], "page": f["page"], "image": f["image"], "file": f["file"],
                          "figure_type": f.get("vlm", {}).get("figure_type", "other"),
                          "caption": f.get("pdf_caption") or f.get("vlm", {}).get("title", ""),
                          "concepts": f["concepts"]})
        mm_pdf.write_figures_ttl(figs, config.ROOT / "data" / "graphs" / "documents.ttl")
        print("figure annotations written to data/graphs/documents.ttl "
              "(load with: python graphdb/load_graphdb.py --only documents)")
    for e in extra_chunks or []:
        items.append({"type": "text", "doc": None, "page": None, **e})
    if not items:
        print("RAG: no documents found in qa/rag_docs – general-knowledge answers will use ontology facts only")
        return 0
    vecs = np.asarray(llm.embed([it["text"] for it in items]), dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FILE.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    np.save(DOC_VEC, vecs)
    n_fig = sum(1 for i in items if i["type"] == "figure")
    n_tab = sum(1 for i in items if i["type"] == "table")
    print(f"index: {len(items) - n_fig - n_tab} text chunks, {n_tab} tables, {n_fig} figures")
    return len(items)


_doc_cache = None


def _load_index():
    global _doc_cache
    if _doc_cache is None:
        if not DOC_FILE.exists():
            _doc_cache = ([], None)
        else:
            items = [json.loads(l) for l in DOC_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
            for it in items:
                it.setdefault("type", "text")
            _doc_cache = (items, np.load(DOC_VEC))
    return _doc_cache


def _scores(question):
    items, vecs = _load_index()
    if not items:
        return items, None
    q = np.asarray(llm.embed(question)[0], dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-9)
    return items, vecs @ q


def retrieve(question, k=None):
    """Text and table passages only (backward compatible)."""
    p, _ = retrieve_mm(question, k_text=k, k_fig=0)
    return p


def retrieve_mm(question, k_text=None, k_fig=None, min_fig_score=None):
    """Multimodal retrieval: (passages, figures). Figures have their own quota so long text documents
    cannot crowd them out, and a minimum similarity so an unrelated figure is never shown.
    Figures tagged with an ontology concept that the question also mentions get a small boost."""
    items, scores = _scores(question)
    if not items:
        return [], []
    k_text = config.RAG_TOP_K if k_text is None else k_text
    k_fig = config.RAG_TOP_K_FIG if k_fig is None else k_fig
    min_fig = config.FIG_MIN_SCORE if min_fig_score is None else min_fig_score
    q_concepts = {str(i) for i, _ in match_concepts(question)} if k_fig else set()
    passages, figures = [], []
    for i in np.argsort(-scores):
        it, sc = items[int(i)], float(scores[int(i)])
        if it["type"] == "figure":
            boost = 0.03 * len(q_concepts & {c["iri"] for c in it.get("concepts", [])})
            if len(figures) < k_fig and sc + boost >= min_fig:
                figures.append({**it, "score": sc + boost})
        elif len(passages) < k_text:
            passages.append({**it, "score": sc})
        if len(passages) >= k_text and len(figures) >= k_fig:
            break
    figures.sort(key=lambda f: -f["score"])
    return passages, figures


# ------------------------------------------------------------------ ontology facts
_onto = None
REL_LABELS = {
    "mitigatedBy": "is mitigated by", "typicalFireImpact": "typical fire impact",
    "providesService": "provides ecosystem service", "suggestedRestoration": "suggested restoration",
    "requiredPPE": "requires PPE",
}


def _load_ontology():
    global _onto
    if _onto is not None:
        return _onto
    from rdflib import Graph, Namespace, RDFS
    SKOS = Namespace(config.PREFIXES["skos"])
    FFO = Namespace(config.PREFIXES["ffo"])
    g = Graph()
    for f in ("ffonto-core.ttl", "ffonto-vocab.ttl"):
        g.parse(config.ONTOLOGY_DIR / f, format="turtle")
    label = {}
    terms = []                                   # (keyword, iri)
    for s, o in list(g.subject_objects(SKOS.prefLabel)) + list(g.subject_objects(RDFS.label)):
        label.setdefault(s, str(o))
        terms.append((str(o).lower(), s))
    for s, o in g.subject_objects(SKOS.hiddenLabel):
        terms.append((str(o).lower(), s))
    _onto = {"g": g, "label": label, "terms": terms, "sorted_terms": sorted(terms, key=lambda t: -len(t[0])), "SKOS": SKOS, "FFO": FFO, "RDFS": RDFS}
    return _onto


def match_concepts(text, max_concepts=6):
    """Ontology concepts (IRI, label) whose label or keyword occurs in the text (longest match first)."""
    try:
        o = _load_ontology()
    except Exception:
        return []
    norm = " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()) + " "
    hits = []
    for kw, iri in o["sorted_terms"]:
        k = " " + re.sub(r"[^a-z0-9]+", " ", kw).strip() + " "
        if len(k.strip()) >= 3 and k in norm and iri not in [h for h, _ in hits]:
            hits.append((iri, o["label"].get(iri, str(iri))))
        if len(hits) >= max_concepts:
            break
    return hits


def ontology_facts(question, max_concepts=6):
    """Find ontology concepts mentioned in the question and return short facts about them."""
    try:
        o = _load_ontology()
    except Exception as e:                         # rdflib missing etc.
        return [f"(ontology facts unavailable: {e})"]
    g, label, SKOS, FFO, RDFS = o["g"], o["label"], o["SKOS"], o["FFO"], o["RDFS"]
    facts = []
    for iri, name in match_concepts(question, max_concepts):
        for d in list(g.objects(iri, SKOS.definition)) + list(g.objects(iri, RDFS.comment)):
            facts.append(f"{name}: {d}")
        for b in g.objects(iri, SKOS.broader):
            facts.append(f"{name} is a kind of {label.get(b, b)}")
        for p, rl in REL_LABELS.items():
            vals = [label.get(v, str(v)) for v in g.objects(iri, FFO[p])]
            if vals:
                facts.append(f"{name} {rl}: {', '.join(vals)}")
    return facts
