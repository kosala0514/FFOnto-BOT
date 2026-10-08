"""
MCP server exposing the knowledge graph as tools (optional).

    python -m qa.mcp_server            # stdio transport, for MCP clients (Open WebUI via mcpo, Claude Desktop, ...)

Tools: list_cqs, run_cq, sparql_select, ontology_lookup, search_documents, ask
"""
from mcp.server.fastmcp import FastMCP
from . import cqs, kg, rag, pipeline

mcp = FastMCP("ffonto-kg")


@mcp.tool()
def list_cqs() -> list:
    """List the competency questions (tested SPARQL templates) the knowledge graph can answer."""
    return [{"id": c.id, "title": c.title, "examples": c.examples[:3], "filters": sorted(c.focus)}
            for c in cqs.REGISTRY.values()]


@mcp.tool()
def run_cq(cq_id: str, record: str = "", event: str = "", site: str = "", date: str = "") -> dict:
    """Run competency question cq_id, optionally narrowed to a record IRI, event code, site name or date (YYYY-MM-DD)."""
    res = cqs.run_cq(cq_id, record=record or None, event=event or None, site=site or None, date=date or None)
    res["rows"] = res["rows"][:100]
    return res


@mcp.tool()
def sparql_select(query: str) -> dict:
    """Run a read-only SPARQL SELECT/ASK query against the FFOnto knowledge graph (a LIMIT is enforced)."""
    q = kg.make_safe(query)
    cols, rows = kg.select(q)
    return {"query": q, "columns": cols, "rows": rows[:100]}


@mcp.tool()
def ontology_lookup(text: str) -> list:
    """Return FFOnto definitions and relations for the concepts mentioned in the text."""
    return rag.ontology_facts(text)


@mcp.tool()
def search_documents(query: str, k_text: int = 5, k_fig: int = 3) -> dict:
    """Multimodal search over the indexed documents: text chunks, tables and figures (with their PNG path,
    the vision model's description and the ontology concepts they show)."""
    passages, figures = rag.retrieve_mm(query, k_text=k_text, k_fig=k_fig)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("concepts",)}
    return {"passages": [strip(p) for p in passages], "figures": [strip(f) for f in figures]}


@mcp.tool()
def ask(question: str, record: str = "") -> dict:
    """Full pipeline: route, query the knowledge graph (or RAG) and answer in natural language with evidence."""
    return pipeline.answer(question, record=record or None)


if __name__ == "__main__":
    mcp.run()
