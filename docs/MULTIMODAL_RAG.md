# Multimodal RAG: text, tables and figures, with visual answers

Plain RAG reads only the text layer of a PDF. A chart of FWI over a season, a fire-behaviour diagram, a risk map or
a PPE infographic is lost, and so is a table whose layout pypdf flattens. The multimodal pipeline keeps all three, and
every answer can come back **with pictures**:

| Answer route | Visual output | Drawn from |
|---|---|---|
| Knowledge graph (CQ or text-to-SPARQL) | Vega-Lite chart (scores vs. thresholds, FWI time series, observed vs. danger threshold, generic bar/line) | the **same SPARQL rows** the text answer used |
| General knowledge (RAG) | the document **figures the answer cites** `[F1]`; uncited but related figures in a collapsed section | the PDF crops, plus the vision model's reading of them |

## How it works

```
INDEX TIME (python -m qa.build_index --docs-only)
  PDF ─ PyMuPDF ─┬─ page text ───────────────► chunks                       type=text
                 ├─ find_tables() ───────────► markdown tables              type=table
                 ├─ embedded raster images ─┐
                 └─ vector-drawing clusters ┴► PNG crop + printed caption + nearby text
                                                │
                              qwen2.5vl:7b ─────┘  JSON: figure_type, title, description,
                              (cached by image hash) key_findings, data_table (chart values), visible_text
                                                │
                              ontology concept tagging (ffv: labels/keywords) ─► data/graphs/documents.ttl
                                                │
                    all items as text ─► bge-m3 ─► ONE vector index (qa/index/rag_chunks.*)

QUESTION TIME (route = rag)
  question ─► top-5 text/table passages + top-3 figures (own quota, min. similarity, concept boost)
           ─► VISUAL RE-READ: the vision model looks at the top-2 figure PNGs again WITH the question
              (the index description was written without knowing what would be asked)
           ─► answer LLM: ontology facts [O] + passages [1..] + figures [F1..] (description, data, reading)
           ─► answer text  +  visuals: cited figures (PNG) first, related ones collapsed
```

**Why "describe, then embed" and not image embeddings?** Text, tables and figures share one embedding space
(bge-m3), so one retriever and one score handle everything. The text-only answer LLM can read figure evidence,
and the stored description is inspectable and citable. Two weaknesses remain. A description can miss a detail, and
the visual re-read at question time is there to fix that. Visually-rich pages with little text also fit this method
less well; for those, see *Upgrade options*.

**Vector charts.** Many PDF charts are not images: they are drawn with vector paths. The extractor clusters the
drawing operations on a page. A large cluster becomes a figure crop, with a margin so axis labels and the legend are
included. Tables found by `find_tables()` are excluded from the figure search because they are indexed as markdown,
which is more accurate than reading them from pixels.

**Link to the ontology.** Every figure is tagged with the FFOnto concepts its description mentions (`dcterms:subject
ffv:CrownFire`, …). These tags go to `data/graphs/documents.ttl`, so figures are also queryable in GraphDB:

```sparql
PREFIX dcterms: <http://purl.org/dc/terms/>  PREFIX ffv: <http://purl.org/ffonto/vocab#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?figure ?label ?type ?hazard WHERE {
  ?figure dcterms:subject ?c ; rdfs:label ?label ; dcterms:type ?type .
  ?c skos:inScheme ffv:HazardScheme ; skos:prefLabel ?hazard .      # figures about any hazard
}
# figures about any kind of vegetation:  ?figure dcterms:subject/skos:broader* ffv:Vegetation
```

## Setup

```bash
pip install -r qa/requirements-qa.txt          # adds pymupdf, Pillow
ollama pull qwen2.5vl:7b                        # vision model (~6 GB at 4-bit); needs a recent Ollama
cp your_guidance.pdf FWI_manual.pdf qa/rag_docs/
python -m qa.build_index --docs-only            # extracts + describes figures (slow once, then cached)
python graphdb/load_graphdb.py --only documents # optional: figure tags into GraphDB
uvicorn qa.api:app --port 8000 ; streamlit run qa/ui_streamlit.py
```

Check `http://localhost:8000/health`. It shows `rag_index: {"text": …, "table": …, "figure": …}` and whether PyMuPDF
is installed. Figure PNGs are in `qa/index/figures/` and are served at `/figures/<file>`.

**16 GB GPU.** `qwen2.5:14b` (about 9 GB) and `qwen2.5vl:7b` (about 6 GB) together come close to 16 GB. If Ollama swaps
them, each RAG answer with figures takes a few seconds longer. Ways to reduce this:

- `VISUAL_REREAD=0` skips the question-time look and uses only the stored descriptions.
- `N_REREAD=1` re-reads only the top figure.
- `export LLM_MODEL=qwen2.5:7b` uses a smaller answer model.

Index time is a one-off batch, so it is not a concern.

| Setting (env var) | Default | Meaning |
|---|---|---|
| `VISION_MODEL` | `qwen2.5vl:7b` | alternatives: `gemma3:12b`, `llama3.2-vision:11b`, `minicpm-v:8b` |
| `RAG_TOP_K_FIG` | 3 | figures retrieved per question |
| `FIG_MIN_SCORE` | 0.45 | cosine below this, the figure is not shown at all (tune on your documents) |
| `VISUAL_REREAD`, `N_REREAD` | 1, 2 | question-specific look at the top figures |
| `MIN_FIG_PX` | 150 | smaller images (icons, logos) are skipped; repeated images are kept only once |
| `ENABLE_KG_CHARTS` | 1 | charts for knowledge-graph answers |

## What the API returns

`POST /chat` now includes `visuals`:

```json
{"answer": "The FWI rose to about 45 by late August, as Figure [F1] shows [1].",
 "visuals": [{"kind": "image", "ref": "F1", "cited": true, "url": "/figures/fwi_manual_p12_1.png",
              "source": "FWI_manual.pdf p.12 (figure)", "figure_type": "chart",
              "reading": "The FWI line peaks at about 45 at the end of August."}],
 "sources": [{"type": "figure", "ref": "F1", "cited": true, "text": "Chart from FWI_manual.pdf …"}]}
```

A knowledge-graph answer returns `{"kind": "vega", "title": …, "spec": {…Vega-Lite v5…}, "source": "knowledge graph (CQ09)"}`.
The MCP server has a new `search_documents` tool, which returns passages and figures.

## Evaluation (ablation for the paper)

```bash
python -m qa.evaluate figures --make-template   # qa/eval/figure_questions.csv lists every indexed figure
# write 1-3 questions + a short reference answer for the useful figures (aim for 50-100 questions)
python -m qa.evaluate figures --judge           # results/eval/figures_summary.md
```

| Measures | Meaning |
|---|---|
| Figure Recall@1, Recall@3, MRR | is the right figure (document + page) retrieved? |
| text-only RAG vs multimodal RAG: token-F1, judge score, McNemar | does adding figures improve answers to figure-dependent questions? |
| cites the gold figure / any figure | does the answer really use the figure it shows the user? (visual faithfulness) |

Keep the questions honest. Include some questions that the text also answers; there, multimodal should not be worse.
Write the questions **before** looking at the system's answers.

## Upgrade options (if Recall@3 is low)

1. **Visual page retrieval (ColPali / ColQwen2).** These models embed page *images* directly (late interaction),
   which is strong for visually-rich pages with little text. Use them as a second retriever next to bge-m3 and
   merge the two rankings with reciprocal-rank fusion. They need `colpali-engine` with about 4–6 GB of VRAM at
   index time.
2. **Chart-to-table models** (DePlot, or the VLM in a stricter extraction mode). When charts carry exact numbers
   the user will ask about, store the extracted table as its own `table` item.
3. **Scanned PDFs** have no text layer. Every page becomes a full-page image figure and the vision model's
   description makes it searchable. For long scanned documents, run OCR first (`ocrmypdf`).

## Limitations

- Values read from charts are approximate, and the prompts make the answer say "about". Do not use chart readings
  where exact thresholds matter: the danger rules use `danger-rules.ttl`, never a chart.
- Figure descriptions are model output. Spot-check `qa/index/figure_descriptions.jsonl`; a wrong description is
  fixed by editing that line, because the cache is reused on the next build.
- Captions are found by position ("Figure 3 …" just below or above the crop). Unusual layouts can miss them.
