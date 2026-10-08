# Natural-language question answering over the FFOnto knowledge graph

Fully local: **Ollama** (LLMs) + **GraphDB** (knowledge graph) + **FastAPI** (orchestrator) + **Streamlit** (chat UI).
The **MCP** server is optional.

```
Streamlit chat UI ──► FastAPI /chat ──► pipeline.answer()
  (pick image/audio)                     │
                                         ├─ 1. ROUTER (LLM, JSON output) – helped by the 8 most similar
                                         │     labelled ForestFire-VQA questions (embedding search)
                                         │       route = kg | rag | chitchat ;  cq = CQ01…CQ26 | NONE
                                         │
                                         ├─ 2a. kg + CQ   ─► tested query queries/CQxx.rq, narrowed to the
                                         │                    selected record / event (SPARQL VALUES) ─► GraphDB
                                         ├─ 2b. kg + NONE ─► text-to-SPARQL (schema + similar CQ queries),
                                         │                    safety check, 1 repair attempt ─► GraphDB
                                         ├─ 2c. rag       ─► ontology facts (definitions, hazard→action, …)
                                         │                    + passages from qa/rag_docs ─► cited answer
                                         │
                                         └─ 3. ANSWER (LLM) – uses ONLY the returned rows / facts / passages
Every reply includes its evidence: route, CQ, SPARQL, rows, sources.
```

## How answers stay tied to the ontology

1. **Most questions never get free-form SPARQL.** The LLM only *chooses* a competency question. The SPARQL that runs is one of your 27 tested queries (59/59 tests pass).
2. **The context is fixed by the UI, not guessed.** The selected image or audio record is injected as `VALUES ?record { <iri> }`. Event-level CQs (hazards, actions) use that record's `ffo:aboutEvent`.
3. **The answer prompt only allows the rows.** If the rows are empty, the assistant says the knowledge graph has no evidence. It does not answer from the LLM's own knowledge.
4. **Generated SPARQL is the exception and is checked.** Only SELECT/ASK is allowed, a LIMIT is enforced, the syntax is validated, and the query uses only ontology terms (the schema summary is built from `ffonto-core.ttl`).
5. **General knowledge is grounded too.** RAG first collects facts about the concepts mentioned in the question from `ffonto-core.ttl` and `ffonto-vocab.ttl`, then adds document passages. Citations distinguish `[O]` (ontology) from `[1]`, `[2]` (documents).

## Setup (RTX 4070 Ti SUPER, 16 GB)

```bash
# 1. Ollama + models (one time)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:14b          # router + answers (~9 GB at 4-bit, fits 16 GB)
ollama pull bge-m3               # embeddings (example search + RAG)
# optional, for the text-to-SPARQL fallback:
ollama pull qwen2.5-coder:14b    # then: export SPARQL_MODEL=qwen2.5-coder:14b

# 2. Python packages
pip install -r qa/requirements-qa.txt

# 3. GraphDB running with the ffonto repository loaded (docs/GRAPHDB.md)

# 4. Build the indexes (from the ffonto-v2 folder)
python -m qa.build_index                 # labelled questions + qa/rag_docs/*
```

The defaults are in `qa/config.py`. Override any of them with an environment variable, e.g. `export LLM_MODEL=qwen3:14b`, `GRAPHDB_REPO=ffonto`.

Model choices for 16 GB of VRAM:

| Role | Default | Alternatives |
|---|---|---|
| Router + answers | `qwen2.5:14b` | `qwen3:14b`, `llama3.1:8b` (faster, a little less accurate) |
| Text-to-SPARQL fallback | same model | `qwen2.5-coder:14b` |
| Embeddings | `bge-m3` | `nomic-embed-text` |

Using two different 14B models makes Ollama swap them in and out of GPU memory. One model for everything is the fastest option.

## Run

```bash
uvicorn qa.api:app --port 8000           # terminal 1 – API   (check http://localhost:8000/health)
streamlit run qa/ui_streamlit.py         # terminal 2 – chat UI (http://localhost:8501)
```

In the UI:

1. Choose an image or audio record in the sidebar.
2. Ask, for example: *"Are the firefighters wearing gloves?"*, *"What hazards are present?"*, *"Is the smoke score above the threshold?"*, *"What was the FWI at Bejaia Region on 2012-06-07?"*, or *"Explain the fire triangle."*
3. Open **Evidence** to see the CQ, SPARQL and rows behind each answer.

API only:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"What hazards are present?","record":"http://purl.org/ffonto/data/record/00000000-0000-4000-8000-000000000081"}' | python -m json.tool
```

## MCP (optional)

```bash
python -m qa.mcp_server
```

This exposes the tools `list_cqs`, `run_cq`, `sparql_select`, `ontology_lookup` and `ask` over stdio, for any MCP client: Claude Desktop, or Open WebUI through `mcpo`. Use it if you want to show an agent working with the knowledge graph through a standard tool protocol. The FastAPI app does not need it.

## Evaluation (paper section)

```bash
python -m qa.build_index --holdout 0.2         # keep 20 % of the labelled questions unseen
python -m qa.evaluate --limit 200              # stage 1: routing + CQ-selection accuracy
python -m qa.evaluate --end-to-end --judge     # stage 2: answers vs ForestFire-VQA answers
```

- **Stage 1** needs no scene data. It reports routing accuracy, CQ-selection accuracy and the most frequent confusions (`results/qa_eval_routing.csv`). The labels come from `docs/FFVQA_CQ_mapping.xlsx`. They were assigned automatically, so review the confusions: some will be labelling errors, not model errors.
- **Stage 2** runs only for held-out questions whose image or audio is in GraphDB. It needs real records and scene graphs. The `--judge` grades from the local LLM are a convenience; check a sample by hand.
- **Baseline comparison:** run stage 2 twice: normally, then with `FORCE_TEXT2SPARQL=1` (the LLM writes all SPARQL itself; the CQ templates are ignored). This gives the "CQ templates vs free text-to-SPARQL" comparison table.

## Before you rely on it

- **Data:** the answers can only be as good as the knowledge graph. Remove the `fixture_*` records and example scenes from `data/`, and add real VLM scene graphs (next step).
- **RAG documents:** `qa/rag_docs/` is empty. Add trusted wildfire guidance documents. `--gk-answers` indexes the dataset's own general-knowledge answers. That is fine for a demo, but **not** for evaluating RAG on those same questions.
- **Follow-up questions** ("and what about the house?") aren't supported yet: each message is answered on its own. Add the previous turn to the router prompt when you need it.
