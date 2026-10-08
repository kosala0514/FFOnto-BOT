"""
FastAPI service.

    uvicorn qa.api:app --port 8000          (run from the ffonto-v2 folder)

POST /chat     {"message": "...", "record": "<record IRI or null>"}  -> answer + evidence
GET  /records  image/audio records for the UI picker
GET  /cqs      the competency-question catalogue
GET  /health   GraphDB + Ollama status
GET  /figures/<file>   PNG figures cut out of the RAG documents (referenced by answer["visuals"])
"""
from typing import Optional
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import config, kg, cqs, pipeline

app = FastAPI(title="FFOnto KG question answering", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=str(config.FIGURES_DIR)), name="figures")


class ChatIn(BaseModel):
    message: str
    record: Optional[str] = None
    record_label: Optional[str] = None
    force_cq: Optional[str] = None


@app.post("/chat")
def chat(body: ChatIn):
    desc = f"{body.record_label} ({body.record})" if body.record and body.record_label else body.record
    return pipeline.answer(body.message, record=body.record, record_desc=desc, force_cq=body.force_cq)


@app.get("/records")
def records():
    return kg.list_records()


@app.get("/cqs")
def list_cqs():
    return [{"id": c.id, "title": c.title, "examples": c.examples, "focus": sorted(c.focus)} for c in cqs.REGISTRY.values()]


@app.get("/health")
def health():
    status = {}
    try:
        kg.select("ASK { ?s ?p ?o }"); status["graphdb"] = "ok"
    except Exception as e:
        status["graphdb"] = f"error: {e}"
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models", [])]
        status["ollama"] = "ok"
        status["models_present"] = {m: any(n.startswith(m) for n in names)
                                    for m in {config.LLM_MODEL, config.SPARQL_MODEL, config.EMBED_MODEL, config.VISION_MODEL}}
    except Exception as e:
        status["ollama"] = f"error: {e}"
    status["example_index"] = pipeline.example_index() is not None
    from . import rag, mm_pdf
    items, _ = rag._load_index()
    status["rag_index"] = {t: sum(1 for i in items if i.get("type") == t) for t in ("text", "table", "figure")}
    status["pymupdf"] = mm_pdf.available()
    return status
