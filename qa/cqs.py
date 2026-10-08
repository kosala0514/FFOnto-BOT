"""
Competency-question registry.

Each tested query in queries/CQxx.rq becomes a *template*: the orchestrator chooses a CQ and
the context (selected record, event, site, date); the query is narrowed with a trailing
SPARQL VALUES block (record / event) and the result rows are filtered (site / date).
No SPARQL is written by the LLM on this path, so answers come only from validated queries
over the ontology-based knowledge graph.
"""
import re
from dataclasses import dataclass, field
from . import config, kg


@dataclass
class CQ:
    id: str
    title: str
    examples: list
    query: str
    columns: list = field(default_factory=list)

    @property
    def focus(self):
        """Which context variables this query can be narrowed by."""
        return {k for k in ("record", "eventCode", "site", "date") if k in self.columns}


def _projected_columns(query):
    body = "\n".join(l for l in query.splitlines() if not l.lstrip().startswith(("#", "PREFIX")))
    m = re.search(r"(?is)SELECT\s+(?:DISTINCT\s+)?(.*?)\bWHERE\b", body)
    if not m:
        return []
    proj = m.group(1)
    aliases = re.findall(r"(?i)\bAS\s+\?(\w+)\s*\)", proj)
    no_expr = proj
    while True:                                             # strip (…) expressions, innermost first
        new = re.sub(r"\([^()]*\)", " ", no_expr)
        if new == no_expr:
            break
        no_expr = new
    plain = re.findall(r"\?(\w+)", no_expr)
    seen, cols = set(), []
    for v in re.findall(r"\?(\w+)", proj):                  # keep SELECT order
        if (v in plain or v in aliases) and v not in seen:
            seen.add(v); cols.append(v)
    return cols


def load_registry():
    reg = {}
    for p in sorted(config.QUERIES_DIR.glob("CQ*.rq")):
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = lines[0].split(":", 1)[1].strip() if ":" in lines[0] else lines[0]
        examples = [l.strip()[4:].strip() for l in lines if l.strip().startswith("#   - ")]
        reg[p.stem] = CQ(p.stem, title, examples, text, _projected_columns(text))
    return reg


REGISTRY = load_registry()


def catalog_text():
    """Compact CQ list for the router prompt."""
    out = []
    for c in REGISTRY.values():
        ex = "; ".join(c.examples[:3])
        out.append(f"- {c.id}: {c.title}  (e.g. {ex})")
    return "\n".join(out)


def _sparql_literal(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_query(cq, record=None, event=None):
    """Narrow a CQ query with a trailing VALUES block (allowed after the whole SELECT in SPARQL 1.1)."""
    q = cq.query.rstrip()
    if record and "record" in cq.columns:
        q += f"\nVALUES ?record {{ <{record}> }}"
    elif event and "eventCode" in cq.columns:
        q += f"\nVALUES ?eventCode {{ {_sparql_literal(event)} }}"
    return q


def run_cq(cq_id, record=None, event=None, site=None, date=None):
    """Execute a CQ for the given context. Returns dict with query, columns, rows, applied filters."""
    cq = REGISTRY.get(cq_id)
    if cq is None:
        raise KeyError(f"unknown CQ {cq_id}")
    applied = {}
    # if the CQ has no ?record column but has ?eventCode, use the selected record's event
    if record and "record" not in cq.columns and "eventCode" in cq.columns and not event:
        event = kg.event_of(record)
        if event:
            applied["event (from selected record)"] = event
    q = build_query(cq, record=record, event=event)
    if record and "record" in cq.columns:
        applied["record"] = kg.short(record)
    elif event and "eventCode" in cq.columns and "event (from selected record)" not in applied:
        applied["event"] = event
    cols, rows = kg.select(q)
    if site and "site" in cols:
        rows = [r for r in rows if site.lower() in r.get("site", "").lower()]
        applied["site"] = site
    if date and "date" in cols:
        rows = [r for r in rows if r.get("date", "").startswith(date)]
        applied["date"] = date
    return {"cq": cq_id, "title": cq.title, "query": q, "columns": cols, "rows": rows, "filters": applied}
