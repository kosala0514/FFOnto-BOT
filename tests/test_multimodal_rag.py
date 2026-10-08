"""
Multimodal RAG + KG charts, tested offline: Ollama (embeddings, vision model, LLM) and PyMuPDF are mocked.
    pytest -q tests/test_multimodal_rag.py
"""
import json, sys, tempfile, types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("requests")


class Rect:
    def __init__(s, *a):
        if len(a) == 1: a = tuple(a[0])
        s.x0, s.y0, s.x1, s.y1 = map(float, a)
    def __iter__(s): return iter((s.x0, s.y0, s.x1, s.y1))
    width = property(lambda s: max(0, s.x1 - s.x0)); height = property(lambda s: max(0, s.y1 - s.y0))
    is_empty = property(lambda s: s.x1 <= s.x0 or s.y1 <= s.y0)
    def __add__(s, t): return Rect(s.x0 + t[0], s.y0 + t[1], s.x1 + t[2], s.y1 + t[3])
    def __or__(s, o): return Rect(min(s.x0, o.x0), min(s.y0, o.y0), max(s.x1, o.x1), max(s.y1, o.y1))
    def __and__(s, o): return Rect(max(s.x0, o.x0), max(s.y0, o.y0), min(s.x1, o.x1), min(s.y1, o.y1))
    def intersects(s, o): return not (s & o).is_empty
    def get_area(s): return 0 if s.is_empty else s.width * s.height
    def contains(s, o): return s.x0 <= o.x0 and s.y0 <= o.y0 and s.x1 >= o.x1 and s.y1 >= o.y1

class Pix:
    def __init__(s, clip): s.clip = clip
    def tobytes(s, fmt): return ("PNG" + str(tuple(s.clip))).encode()

class Page:
    def __init__(s, n): s.n = n; s.rect = Rect(0, 0, 600, 800)
    def get_text(s, kind="text"):
        if kind == "text": return f"Page {s.n} text about crown fire.\n\nSecond paragraph."
        return [(50, 420, 550, 440, "Figure 2. FWI over August 2012", 0, 0), (50, 450, 550, 500, "The FWI rose sharply.", 1, 0),
                (50, 20, 550, 40, "Header", 2, 0)]
    def find_tables(s): raise RuntimeError("no tables in mock")
    def get_images(s, full=True): return [(7, 0, 800, 600, 8, "DeviceRGB", "", "Im1", "DCTDecode", 0), (8,) + (0,) * 9]
    def get_image_rects(s, xref):
        return [Rect(60, 60, 540, 400)] if xref == 7 else [Rect(10, 10, 30, 30)]   # 2nd = logo/icon
    def get_drawings(s):
        if s.n != 1: return []
        return [{"rect": Rect(100 + i * 10, 500, 105 + i * 10, 700)} for i in range(30)] + [{"rect": Rect(0, 780, 600, 781)}]
    def cluster_drawings(s, **kw): raise AttributeError("old PyMuPDF")
    def get_pixmap(s, dpi=150, clip=None): return Pix(clip)

class Doc:
    def __iter__(s): return iter([Page(0), Page(1)])
    def close(s): pass



def test_pdf_extraction_with_mock_pymupdf(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymupdf", types.SimpleNamespace(Rect=Rect, open=lambda p: Doc()))
    from qa import mm_pdf
    tmp = Path(tempfile.mkdtemp())
    texts, tables, figs = mm_pdf.extract_pdf(tmp / "guide.pdf", out_dir=tmp)
    assert len(texts) == 2
    assert [f["kind"] for f in figs] == ["raster", "vector"]      # logo skipped, duplicate image skipped
    assert figs[0]["pdf_caption"].startswith("Figure 2")
    assert all((tmp / f["file"]).exists() for f in figs)


def test_index_retrieve_answer_and_charts(monkeypatch):
    from qa import config, llm, rag, mm_pdf, pipeline, visuals
    monkeypatch.setattr(mm_pdf, "available", lambda: False)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "docs").mkdir()
    for k, v in dict(RAG_DOCS_DIR=tmp / "docs", INDEX_DIR=tmp / "index", FIGURES_DIR=tmp / "index/figures",
                     FIGURE_CACHE=tmp / "index/fig.jsonl", ROOT=tmp).items():
        monkeypatch.setattr(config, k, v)
    monkeypatch.setattr(rag, "DOC_FILE", tmp / "index/rag_chunks.jsonl")
    monkeypatch.setattr(rag, "DOC_VEC", tmp / "index/rag_chunks.npy")
    monkeypatch.setattr(rag, "_doc_cache", None)
    (config.RAG_DOCS_DIR / "guide.md").write_text("Crown fire spreads through tree canopies.\n\nEvacuate early when FWI is extreme.")
    Image.new("RGB", (400, 300), "orange").save(config.RAG_DOCS_DIR / "fwi_chart.png")
    Image.new("RGB", (400, 300), "green").save(config.RAG_DOCS_DIR / "forest_photo.png")

    VOC = ["crown fire", "fwi", "evacuate", "chart", "canopy", "photo", "forest", "trend"]
    def fake_embed(texts, model=None):
        if isinstance(texts, str): texts = [texts]
        return [[t.lower().count(w) + 0.01 for w in VOC] for t in texts]
    calls = {"vision": 0}
    def fake_vision(system, user, images, schema=None, model=None, temperature=0):
        calls["vision"] += 1
        name = Path(images[0]).name
        if schema:
            if "fwi" in name:
                return {"figure_type": "chart", "title": "FWI trend chart", "description": "Line chart of FWI rising over August; crown fire risk.",
                        "key_findings": ["FWI peaks at ~45"], "data_table": "| date | FWI |\n|---|---|\n| Aug 1 | ~10 |", "visible_text": "FWI"}
            return {"figure_type": "photo", "title": "Forest photo", "description": "A green forest.", "key_findings": [], "data_table": "", "visible_text": ""}
        return "The chart shows FWI rising to about 45." if "fwi" in name else "NOT RELEVANT"
    monkeypatch.setattr(llm, 'embed', fake_embed); monkeypatch.setattr(llm, 'chat_vision', fake_vision)
    monkeypatch.setattr(rag, "ontology_facts", lambda q: [])
    monkeypatch.setattr(rag, 'match_concepts', lambda text, max_concepts=6: [("http://purl.org/ffonto/vocab#CrownFire", "Crown fire")] if "crown fire" in text.lower() else [])

    n = rag.build_doc_index()
    items = [json.loads(l) for l in rag.DOC_FILE.read_text().splitlines()]
    print("items", n, [i["type"] for i in items])
    assert {i["type"] for i in items} == {"text", "figure"}
    # rebuild uses cache
    assert (tmp / 'data/graphs/documents.ttl').exists()
    v0 = calls["vision"]; rag._doc_cache = None; rag.build_doc_index(); assert calls["vision"] == v0, "cache not used"

    rag._doc_cache = None
    p, f = rag.retrieve_mm("What does the FWI trend chart show?")
    print("passages", [x["source"] for x in p]); print("figures", [(x["source"], round(x["score"], 2)) for x in f])
    assert f and "fwi_chart" in f[0]["file"]

    monkeypatch.setattr(llm, "chat_text", lambda system, user, model=None, temperature=0.2: "FWI rose to about 45 in August, as Figure [F1] shows [1].")
    ans, facts, passages, figs = pipeline.rag_answer("What does the FWI trend chart show?")
    vis = pipeline.figure_visuals(figs)
    print(ans); print(json.dumps(vis, indent=1)[:600])
    assert vis[0]["cited"] and vis[0]["reading"]

    # full answer() through rag route
    monkeypatch.setattr(pipeline, "route", lambda q, d: {"route": "rag", "cq": "NONE", "examples": []})
    out = pipeline.answer("What does the FWI trend chart show?")
    print("answer() visuals:", [(v["kind"], v["cited"]) for v in out["visuals"]], "error:", out["error"])
    assert out["visuals"] and not out["error"]

    # KG charts
    r05 = [{"record": "ffd:record/a", "class": "ffv:Fire", "confidence": "0.16", "threshold": "0.55", "aboveThreshold": "false"},
           {"record": "ffd:record/a", "class": "ffv:Smoke", "confidence": "0.73", "threshold": "0.6", "aboveThreshold": "true"}]
    r09 = [{"w": "ffd:w/1", "site": "ffd:site/montesinho/grid-7-5", "date": f"2012-08-{d:02d}T00:00:00Z", "FFMC": "90", "DMC": str(100+d),
            "DC": "600", "ISI": "8", "BUI": "120", "FWI": str(20+d), "previousDate": "", "previousFWI": ""} for d in (26, 27, 28)]
    r09s = [{**r09[2], "previousDate": "2012-08-27", "previousFWI": "47"}]
    r10 = [{"w": "x", "site": "s", "date": "2012-08-28", "condition": "ffv:HighFWI", "ruleLabel": "Elevated fire danger", "property": "ffo:fwi",
            "operator": ">=", "threshold": "30.0", "observedValue": "33.1"}]
    rg = [{"kind": "ffv:Firefighter", "count": "3"}, {"kind": "ffv:Vehicle", "count": "1"}]
    for cq, cols, rows in (("CQ05", list(r05[0]), r05), ("CQ09", list(r09[0]), r09), ("CQ09", list(r09s[0]), r09s),
                           ("CQ10", list(r10[0]), r10), ("CQ13", ["kind", "count"], rg), ("CQ02", ["record", "missing"], [{"record": "x", "missing": "crs"}])):
        ch = visuals.chart_for(cq, cols, rows, cq)
        for c in ch: json.dumps(c)
        assert bool(ch) == (cq != "CQ02"), cq
