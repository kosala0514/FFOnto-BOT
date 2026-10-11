"""GraphDB access: run SELECT queries, list records, safety checks for generated SPARQL."""
import re
import requests
from . import config

_NS = sorted(config.PREFIXES.items(), key=lambda kv: -len(kv[1]))


class KGError(RuntimeError):
    pass


def short(value):
    """Shorten IRIs with known prefixes; data IRIs keep only the readable tail."""
    if not isinstance(value, str):
        return value
    for p, ns in _NS:
        if value.startswith(ns):
            local = value[len(ns):]
            if p == "ffd":                      # e.g. record/3f6d…/obs/0 -> keep it short but unique
                return "ffd:" + local
            return f"{p}:{local}"
    return value


def select(query, timeout=60):
    """Run a SELECT query. Returns (columns, rows) where rows are dicts of display strings."""
    url = f"{config.GRAPHDB_URL}/repositories/{config.GRAPHDB_REPO}"
    try:
        r = requests.post(url, data={"query": query}, timeout=timeout,
                          headers={"Accept": "application/sparql-results+json"})
    except requests.RequestException as e:
        raise KGError(f"GraphDB not reachable at {url}: {e}")
    if r.status_code >= 300:
        raise KGError(f"GraphDB error {r.status_code}: {r.text[:500]}")
    data = r.json()
    if "boolean" in data:                       # ASK
        return ["answer"], [{"answer": str(data["boolean"])}]
    cols = data["head"]["vars"]
    rows = []
    for b in data["results"]["bindings"]:
        row = {}
        for c in cols:
            if c in b:
                v = b[c]
                row[c] = short(v["value"]) if v["type"] == "uri" else v["value"]
            else:
                row[c] = ""
        rows.append(row)
    return cols, rows


# ----------------------------------------------------------------------------- records (for the UI)
RECORDS_QUERY = config.PREFIX_BLOCK + """
SELECT ?record ?type ?label ?event ?file WHERE {
  VALUES ?type { ffonto:ImageRecord ffonto:AudioRecord }
  ?record a ?type .
  OPTIONAL { ?record ffonto:mediaId ?m }
  OPTIONAL { ?record ffonto:fileURI ?file }
  OPTIONAL { ?record ffonto:aboutEvent/ffonto:eventCode ?event }
  BIND(COALESCE(?m, REPLACE(STR(?file), "^.*/", ""), STR(?record)) AS ?label)
} ORDER BY ?type ?label
"""


def list_records():
    url = f"{config.GRAPHDB_URL}/repositories/{config.GRAPHDB_REPO}"
    r = requests.post(url, data={"query": RECORDS_QUERY}, timeout=60,
                      headers={"Accept": "application/sparql-results+json"})
    r.raise_for_status()
    out = []
    for b in r.json()["results"]["bindings"]:
        out.append({"iri": b["record"]["value"],
                    "type": b["type"]["value"].rsplit("#", 1)[-1],
                    "label": b.get("label", {}).get("value", ""),
                    "event": b.get("event", {}).get("value", ""),
                    "file": b.get("file", {}).get("value", "")})
    return out


def event_of(record_iri):
    q = config.PREFIX_BLOCK + f"\nSELECT ?e WHERE {{ <{record_iri}> ffonto:aboutEvent/ffonto:eventCode ?e }} LIMIT 1"
    _, rows = select(q)
    return rows[0]["e"] if rows else None


# ----------------------------------------------------------------------------- safety for generated SPARQL
FORBIDDEN = re.compile(r"\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|ADD|MOVE|COPY|SERVICE)\b", re.I)


def make_safe(query):
    """Add missing prefixes, allow only SELECT/ASK, enforce a LIMIT, check syntax with rdflib if available."""
    q = query.strip().strip("`")
    q = re.sub(r"^sparql\s*", "", q, flags=re.I)
    if FORBIDDEN.search(re.sub(r'"[^"]*"', '""', q)):
        raise KGError("only read-only SELECT/ASK queries are allowed")
    body = re.sub(r"(?im)^\s*PREFIX\s+\S+\s*<[^>]*>\s*", "", q)
    if not re.match(r"(?is)^\s*(SELECT|ASK)\b", body):
        raise KGError("query must be a SELECT or ASK query")
    declared = set(re.findall(r"(?im)^\s*PREFIX\s+(\w*):", q))
    missing = [f"PREFIX {p}: <{u}>" for p, u in config.PREFIXES.items() if p not in declared]
    q = "\n".join(missing) + "\n" + q
    if re.match(r"(?is)^\s*SELECT\b", body) and not re.search(r"(?i)\bLIMIT\s+\d+", body):
        q += f"\nLIMIT {config.SPARQL_LIMIT}"
    try:
        from rdflib.plugins.sparql import prepareQuery
        prepareQuery(q)
    except ImportError:
        pass
    except Exception as e:
        raise KGError(f"SPARQL syntax error: {e}")
    return q
