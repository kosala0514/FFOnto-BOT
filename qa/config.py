"""Settings – override any value with an environment variable of the same name."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # ffonto-v2/


def env(name, default):
    return os.environ.get(name, default)


# GraphDB
GRAPHDB_URL = env("GRAPHDB_URL", "http://localhost:7200")
GRAPHDB_REPO = env("GRAPHDB_REPO", "ffonto-kg")

# Ollama (local LLMs). Sized for a 16 GB GPU: a 14B model at 4-bit uses ~9-10 GB.
OLLAMA_URL = env("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = env("LLM_MODEL", "qwen2.5:14b")              # routing, CQ selection, answers
SPARQL_MODEL = env("SPARQL_MODEL", LLM_MODEL)              # free text-to-SPARQL fallback (e.g. qwen2.5-coder:14b)
EMBED_MODEL = env("EMBED_MODEL", "bge-m3")                 # example retrieval + RAG
LLM_TIMEOUT = int(env("LLM_TIMEOUT", "180"))
# Vision-language model for figures/charts in PDFs (multimodal RAG). ~6 GB at 4-bit.
# Alternatives: gemma3:12b, llama3.2-vision:11b, minicpm-v:8b
VISION_MODEL = env("VISION_MODEL", "qwen2.5vl:7b")

# Behaviour
MAX_ROWS_TO_LLM = int(env("MAX_ROWS_TO_LLM", "40"))
SPARQL_LIMIT = int(env("SPARQL_LIMIT", "200"))
N_EXAMPLES = int(env("N_EXAMPLES", "8"))                   # labelled questions shown to the router
RAG_TOP_K = int(env("RAG_TOP_K", "5"))
RAG_TOP_K_FIG = int(env("RAG_TOP_K_FIG", "3"))             # figures retrieved per question (separate quota)
FIG_MIN_SCORE = float(env("FIG_MIN_SCORE", "0.45"))        # cosine below this: figure not relevant enough
VISUAL_REREAD = env("VISUAL_REREAD", "1") == "1"           # let the VLM look at the top figures again for this question
N_REREAD = int(env("N_REREAD", "2"))
MIN_FIG_PX = int(env("MIN_FIG_PX", "150"))                 # skip icons/logos smaller than this (pixels, both sides)
ENABLE_KG_CHARTS = env("ENABLE_KG_CHARTS", "1") == "1"     # Vega-Lite charts from knowledge-graph rows
# When the KG-only answer says "no information" although rows came back, answer again by interpreting the rows
# with ontology facts + document passages (marked "general knowledge"). Off = strict KG-only answers.
ANSWER_FUSION = env("ANSWER_FUSION", "1") == "1"
# Visual check: when the KG has no positive evidence for a yes/no question about an image, let the vision model
# look at the image itself (the KG records only what scene extraction found). Off by default in the chatbot.
VLM_FALLBACK = env("VLM_FALLBACK", "0") == "1"
# Where the dataset images are, if the file paths stored in the KG do not exist on this machine (searched by file name)
IMAGES_DIR = env("IMAGES_DIR", "")
ENABLE_TEXT2SPARQL = env("ENABLE_TEXT2SPARQL", "1") == "1"
FORCE_TEXT2SPARQL = env("FORCE_TEXT2SPARQL", "0") == "1"     # baseline: ignore CQ templates (for the comparison experiment)
# Text-to-SPARQL mode: "naive" = the model writes the whole query (baseline);
# "plus" = the four improvements below (each can be switched off for an ablation study)
T2S_MODE = env("T2S_MODE", "plus")
T2S_BIND_CONTEXT = env("T2S_BIND_CONTEXT", "1") == "1"     # 1. code binds ?record / ?eventCode / ?date, not the model
T2S_SCHEMA_LINK = env("T2S_SCHEMA_LINK", "1") == "1"       # 2. show the vocabulary concepts the question mentions
T2S_ADAPT_TEMPLATE = env("T2S_ADAPT_TEMPLATE", "1") == "1" # 3. start from the nearest tested CQ query and edit it
T2S_EXEC_FEEDBACK = env("T2S_EXEC_FEEDBACK", "1") == "1"   # 4. regenerate once if the result is empty or too large
T2S_MAX_ROWS = int(env("T2S_MAX_ROWS", "40"))              # "too large" for a question about one record / date

# Paths
QUERIES_DIR = ROOT / "queries"
ONTOLOGY_DIR = ROOT / "ontology"
MAPPING_XLSX = ROOT / "docs" / "FFVQA_CQ_mapping.xlsx"
INDEX_DIR = ROOT / "qa" / "index"
RAG_DOCS_DIR = ROOT / "qa" / "rag_docs"
FIGURES_DIR = INDEX_DIR / "figures"                        # PNGs cut out of the PDFs (served by the API at /figures)
FIGURE_CACHE = INDEX_DIR / "figure_descriptions.jsonl"     # VLM descriptions, keyed by image hash (re-runs are cheap)

PREFIXES = {
    "ffonto": "http://purl.org/ffonto#",
    "ffontov": "http://purl.org/ffonto/vocab#",
    "ffd": "http://purl.org/ffonto/data/",
    "sosa": "http://www.w3.org/ns/sosa/",
    "prov": "http://www.w3.org/ns/prov#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "dcterms": "http://purl.org/dc/terms/",
}
PREFIX_BLOCK = "\n".join(f"PREFIX {p}: <{u}>" for p, u in PREFIXES.items())
