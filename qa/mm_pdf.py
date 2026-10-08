"""
Multimodal document ingestion for RAG: text, tables AND figures from PDFs (plus standalone images).

    PDF page ─┬─ text blocks ─────────────► text chunks              (type "text")
              ├─ tables (find_tables) ────► markdown tables          (type "table")
              ├─ raster images ─┐
              └─ vector charts ─┴─► PNG crop ─► VLM description ──► figure items  (type "figure")
                                               (type, description, key findings,
                                                data table, ontology concepts)

Why describe-then-embed?  Every item - text, table or figure - ends up as text in ONE bge-m3 embedding space,
so one retriever handles all three and the answer LLM (text-only) can read the figure evidence. The original
PNG is kept, shown to the user, and (optionally) looked at again by the VLM for the specific question.

Requires PyMuPDF:  pip install pymupdf      (falls back to text-only pypdf extraction if missing)
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

from . import config, llm

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
CAPTION_RE = re.compile(r"^\s*(fig(\.|ure)?|table|chart|graph|map|plate|exhibit)\s*[\dIVX]", re.I)


def _pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError:
            return None


def available():
    return _pymupdf() is not None


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "doc"


def sha1_file(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ geometry helpers (PyMuPDF Rects)
def _merge_rects(rects, tol=12):
    """Greedy merge of rectangles that touch or lie within `tol` points of each other."""
    rects = [r for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        while rects:
            r = rects.pop()
            grown = r + (-tol, -tol, tol, tol)
            i = 0
            while i < len(rects):
                if grown.intersects(rects[i]):
                    r = r | rects.pop(i)
                    grown = r + (-tol, -tol, tol, tol)
                    changed = True
                else:
                    i += 1
            out.append(r)
        rects = out
    return rects


def _vector_figure_rects(page, fitz):
    """Regions with many vector drawing operations = charts / diagrams drawn as vectors (not images)."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    if len(drawings) < 15:                       # a few lines/boxes: page decoration, not a chart
        return []
    pr = page.rect
    rects = []
    try:                                         # PyMuPDF >= 1.24
        rects = page.cluster_drawings(drawings=drawings, x_tolerance=12, y_tolerance=12)
    except Exception:
        raw = [fitz.Rect(d["rect"]) for d in drawings
               if not (d["rect"].width > 0.9 * pr.width and d["rect"].height < 3)]   # skip page-wide rules
        rects = _merge_rects(raw)
    keep = []
    for r in rects:
        r = fitz.Rect(r)
        if r.width >= 0.25 * pr.width and r.height >= 0.12 * pr.height and r.get_area() < 0.95 * pr.get_area():
            keep.append((r + (-20, -14, 20, 24)) & pr)   # a margin to include axis labels / legend text
    return keep


def _overlap(a, b):
    inter = a & b
    if inter.is_empty:
        return 0.0
    return inter.get_area() / max(1e-6, min(a.get_area(), b.get_area()))


def _caption_and_context(page, rect, fitz):
    """Printed caption ('Figure 3. …') just below/above the figure, plus nearby text as context."""
    caption, near = "", []
    for b in page.get_text("blocks"):
        if len(b) > 6 and b[6] != 0:             # image block
            continue
        br, text = fitz.Rect(b[:4]), " ".join(str(b[4]).split())
        if not text:
            continue
        horiz = min(br.x1, rect.x1) - max(br.x0, rect.x0) > 0
        below = 0 <= br.y0 - rect.y1 <= 60
        above = 0 <= rect.y0 - br.y1 <= 45
        if horiz and (below or above) and CAPTION_RE.match(text) and not caption:
            caption = text[:400]
        elif horiz and (0 <= br.y0 - rect.y1 <= 160 or 0 <= rect.y0 - br.y1 <= 160) and not rect.contains(br):
            near.append(text)
    return caption, " ".join(near)[:700]


def _table_markdown(tab):
    try:
        return tab.to_markdown()
    except Exception:
        rows = tab.extract() or []
        if not rows:
            return ""
        clean = lambda c: " ".join(str(c or "").split())
        lines = ["| " + " | ".join(clean(c) for c in rows[0]) + " |", "|" + "---|" * len(rows[0])]
        lines += ["| " + " | ".join(clean(c) for c in r) + " |" for r in rows[1:]]
        return "\n".join(lines)


# ------------------------------------------------------------------ extraction
def extract_pdf(path, out_dir=None, dpi=150, max_side=1280):
    """Return (text_pages, tables, figures) for one PDF. Figures are saved as PNG in out_dir."""
    fitz = _pymupdf()
    if fitz is None:
        raise ImportError("pip install pymupdf")
    path = Path(path)
    out_dir = Path(out_dir or config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(path))
    texts, tables, figures, seen = [], [], [], {}
    for pno, page in enumerate(doc):
        pr = page.rect
        # --- tables first (so their area is not treated as a chart)
        table_rects = []
        try:
            for t_i, tab in enumerate(page.find_tables().tables):
                md = _table_markdown(tab)
                if md and md.count("\n") >= 2:
                    tr = fitz.Rect(tab.bbox)
                    table_rects.append(tr)
                    cap, _ = _caption_and_context(page, tr, fitz)
                    tables.append({"doc": path.name, "page": pno + 1, "caption": cap, "markdown": md})
        except Exception:
            pass
        texts.append({"doc": path.name, "page": pno + 1, "text": page.get_text("text")})

        # --- candidate figure regions: raster images + vector charts
        cands = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for r in rects:
                r = fitz.Rect(r) & pr
                if r.width * dpi / 72 >= config.MIN_FIG_PX and r.height * dpi / 72 >= config.MIN_FIG_PX:
                    cands.append(("raster", r))
        for r in _vector_figure_rects(page, fitz):
            if not any(_overlap(r, c[1]) > 0.5 for c in cands):
                cands.append(("vector", r))
        cands = [c for c in cands if not any(_overlap(c[1], t) > 0.6 for t in table_rects)]

        for k, (kind, r) in enumerate(cands):
            zoom_dpi = min(dpi, max_side * 72 / max(r.width, r.height, 1))
            pix = page.get_pixmap(dpi=int(zoom_dpi), clip=r)
            png = pix.tobytes("png")
            h = hashlib.sha1(png).hexdigest()
            seen[h] = seen.get(h, 0) + 1
            if seen[h] > 1:                      # repeated logo / header image
                continue
            fname = f"{slug(path.stem)}_p{pno + 1}_{k + 1}.png"
            (out_dir / fname).write_bytes(png)
            cap, ctx = _caption_and_context(page, r, fitz)
            figures.append({"doc": path.name, "page": pno + 1, "kind": kind, "bbox": [round(v, 1) for v in r],
                            "image": str(out_dir / fname), "file": fname, "sha1": h,
                            "pdf_caption": cap, "context": ctx})
    doc.close()
    return texts, tables, figures


def standalone_image(path, out_dir=None):
    """A PNG/JPG put directly in qa/rag_docs (e.g. an infographic) is indexed as a figure."""
    path = Path(path)
    out_dir = Path(out_dir or config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{slug(path.stem)}{path.suffix.lower()}"
    shutil.copyfile(path, out_dir / fname)
    return {"doc": path.name, "page": 1, "kind": "image", "bbox": None, "image": str(out_dir / fname),
            "file": fname, "sha1": sha1_file(path), "pdf_caption": "", "context": ""}


# ------------------------------------------------------------------ VLM description (cached)
DESCRIBE_SYSTEM = """You describe one figure taken from a wildfire-related document so that it can be found by search
and used as evidence. Describe ONLY what is visible; never invent numbers or facts.
- chart/graph: say what is plotted (axes, units, series, legend), the ranges and the trend; give notable values.
  If values are readable, transcribe them into a markdown table in data_table (prefix estimates with ~).
- map: the region, legend classes and what is highlighted.
- diagram/infographic: the components and how they connect, in reading order.
- photo: the scene (fire, smoke, flames, vegetation, terrain, people, PPE, vehicles, equipment, damage).
- table: transcribe it into data_table.
Copy readable text (titles, labels, legend) into visible_text. Answer with JSON only."""

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "figure_type": {"type": "string", "enum": ["chart", "map", "diagram", "infographic", "photo", "table", "other"]},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "data_table": {"type": "string"},
        "visible_text": {"type": "string"},
    },
    "required": ["figure_type", "title", "description", "key_findings", "data_table", "visible_text"],
}


def _load_cache():
    cache = {}
    if config.FIGURE_CACHE.exists():
        for line in config.FIGURE_CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["sha1"]] = d
    return cache


def describe_figures(figures, model=None, verbose=True):
    """Add the VLM description to every figure (in place). Descriptions are cached by image hash."""
    cache = _load_cache()
    config.FIGURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with config.FIGURE_CACHE.open("a", encoding="utf-8") as fh:
        for i, f in enumerate(figures):
            if f["sha1"] in cache:
                f["vlm"] = cache[f["sha1"]]["vlm"]
                continue
            user = (f"Document: {f['doc']}, page {f['page']}.\n"
                    f"Caption printed near the figure: {f['pdf_caption'] or '(none)'}\n"
                    f"Text around the figure: {f['context'][:600] or '(none)'}")
            try:
                f["vlm"] = llm.chat_vision(DESCRIBE_SYSTEM, user, [f["image"]], schema=DESCRIBE_SCHEMA, model=model)
            except llm.LLMError as e:
                print(f"  ! {f['file']}: {e}")
                f["vlm"] = {"figure_type": "other", "title": "", "description": "", "key_findings": [],
                            "data_table": "", "visible_text": ""}
                continue                          # not cached: retried next time
            fh.write(json.dumps({"sha1": f["sha1"], "vlm": f["vlm"]}, ensure_ascii=False) + "\n")
            if verbose:
                print(f"  described {i + 1}/{len(figures)} {f['file']}: {f['vlm'].get('figure_type')} – "
                      f"{(f['vlm'].get('title') or f['vlm'].get('description', ''))[:70]}")
    return figures


def figure_text(f):
    """The text that is embedded for a figure (and shown to the answer LLM)."""
    v = f.get("vlm", {})
    parts = [f"{(v.get('figure_type') or 'figure').capitalize()} from {f['doc']} (page {f['page']})."]
    if f.get("pdf_caption"):
        parts.append(f"Caption: {f['pdf_caption']}")
    if v.get("title"):
        parts.append(f"Title: {v['title']}")
    if v.get("description"):
        parts.append(f"Description: {v['description']}")
    if v.get("key_findings"):
        parts.append("Key findings: " + "; ".join(v["key_findings"]))
    if v.get("data_table"):
        parts.append("Data:\n" + v["data_table"][:1500])
    if v.get("visible_text"):
        parts.append(f"Text in figure: {v['visible_text'][:400]}")
    if f.get("concepts"):
        parts.append("Ontology concepts: " + ", ".join(c["label"] for c in f["concepts"]))
    return "\n".join(parts)


# ------------------------------------------------------------------ figures as RDF (optional KG link)
def write_figures_ttl(figures, path):
    """Each figure becomes a prov:Entity tagged with the ontology concepts it shows (dcterms:subject).
    Load it as named graph <http://purl.org/ffonto/graph/documents> to ask e.g. 'which figures show crown fire?'."""
    def lit(s):
        return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    lines = ["@prefix ffd: <http://purl.org/ffonto/data/> .", "@prefix prov: <http://www.w3.org/ns/prov#> .",
             "@prefix dcterms: <http://purl.org/dc/terms/> .", "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", ""]
    for f in figures:
        v = f.get("vlm", {})
        iri = f"<http://purl.org/ffonto/data/doc/{slug(Path(f['doc']).stem)}/figure/{Path(f['file']).stem}>"
        lab = f"{f['doc']} p.{f['page']}: " + (f.get("pdf_caption") or v.get("title") or Path(f["file"]).stem)
        t = [f"{iri} a prov:Entity ;", f"    rdfs:label {lit(lab[:200])} ;",
             f"    dcterms:source {lit(f['doc'])} ;", f"    dcterms:type {lit(v.get('figure_type', 'other'))} ;",
             f"    dcterms:identifier {lit(f['file'])} ;"]
        if v.get("description"):
            t.append(f"    rdfs:comment {lit(v['description'][:1000])} ;")
        for c in f.get("concepts", []):
            t.append(f"    dcterms:subject <{c['iri']}> ;")
        t[-1] = t[-1].rstrip(" ;") + " ."
        lines += t + [""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
