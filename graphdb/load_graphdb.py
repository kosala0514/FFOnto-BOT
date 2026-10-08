#!/usr/bin/env python3
"""
load_graphdb.py — load FFOnto into ONE GraphDB repository, one NAMED GRAPH per source,
apply the FFOnto rules inside GraphDB, and run the CQ queries against the endpoint.

    python scripts/lift.py                                   # produces data/graphs/*.ttl
    python graphdb/load_graphdb.py                           # create repo (if needed), load, rules, CQs
    python graphdb/load_graphdb.py --only image              # reload just the image graph (+ rules)
    python graphdb/load_graphdb.py --url http://host:7200 --repo ffonto --recreate

Named graphs (all queried together, because GraphDB's default graph is the union):
    <http://purl.org/ffonto/graph/ontology>   core schema + danger rules
    <http://purl.org/ffonto/graph/vocab>      SKOS vocabulary + background knowledge
    <http://purl.org/ffonto/graph/image>      image records, classifier observations, captions
    <http://purl.org/ffonto/graph/audio>      audio records and sound-event observations
    <http://purl.org/ffonto/graph/weather>    weather / FWI rows
    <http://purl.org/ffonto/graph/scene>      VLM scene descriptions
    <http://purl.org/ffonto/graph/inferred>   output of rules/R00–R07 (rebuilt on every load)
"""
import argparse, re, sys, time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

ROOT = Path(__file__).resolve().parent.parent
G = "http://purl.org/ffonto/graph/"
SOURCES = {
    "ontology": [ROOT / "ontology/ffonto-core.ttl", ROOT / "ontology/danger-rules.ttl"],
    "vocab":    [ROOT / "ontology/ffonto-vocab.ttl"],
    "image":    [ROOT / "data/graphs/image.ttl"],
    "audio":    [ROOT / "data/graphs/audio.ttl"],
    "weather":  [ROOT / "data/graphs/weather.ttl"],
    "scene":    [ROOT / "data/graphs/scene.ttl"],
}


class GraphDB:
    def __init__(self, url, repo):
        self.url, self.repo = url.rstrip("/"), repo
        self.s = requests.Session()

    def repos(self):
        r = self.s.get(f"{self.url}/rest/repositories", timeout=30)
        r.raise_for_status()
        return [x["id"] for x in r.json()]

    def create(self, config_path):
        cfg = Path(config_path).read_text(encoding="utf-8")
        cfg = re.sub(r'rep:repositoryID "[^"]+"', f'rep:repositoryID "{self.repo}"', cfg)
        r = self.s.post(f"{self.url}/rest/repositories",
                        files={"config": ("repo-config.ttl", cfg.encode("utf-8"), "text/turtle")}, timeout=60)
        if r.status_code >= 300:
            sys.exit(f"could not create repository: {r.status_code} {r.text}")

    def delete(self):
        self.s.delete(f"{self.url}/rest/repositories/{self.repo}", timeout=60)

    def put_graph(self, graph, path, replace=True):
        """RDF4J Graph Store protocol: PUT replaces the named graph, POST adds to it."""
        fn = self.s.put if replace else self.s.post
        r = fn(f"{self.url}/repositories/{self.repo}/rdf-graphs/service", params={"graph": graph},
               data=Path(path).read_bytes(), headers={"Content-Type": "text/turtle"}, timeout=600)
        if r.status_code >= 300:
            sys.exit(f"upload of {path} failed: {r.status_code} {r.text[:500]}")

    def clear_graph(self, graph):
        self.update(f"CLEAR SILENT GRAPH <{graph}>")

    def update(self, sparql):
        r = self.s.post(f"{self.url}/repositories/{self.repo}/statements", data=sparql.encode("utf-8"),
                        headers={"Content-Type": "application/sparql-update"}, timeout=600)
        if r.status_code >= 300:
            raise RuntimeError(f"{r.status_code} {r.text[:800]}")

    def select(self, sparql, accept="application/sparql-results+json"):
        r = self.s.post(f"{self.url}/repositories/{self.repo}", data={"query": sparql},
                        headers={"Accept": accept}, timeout=600)
        if r.status_code >= 300:
            raise RuntimeError(f"{r.status_code} {r.text[:800]}")
        return r.json() if accept.endswith("json") else r.text

    def count(self, graph):
        res = self.select(f"SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}")
        return int(res["results"]["bindings"][0]["n"]["value"])


def construct_to_insert(rule_text, graph):
    """Turn 'PREFIX… CONSTRUCT { T } WHERE { … }' into 'PREFIX… INSERT { GRAPH <g> { T } } WHERE { … }'."""
    i = re.search(r"^CONSTRUCT\b", rule_text, flags=re.M).start()   # skip the word in comments
    j = rule_text.index("\nWHERE", i)
    template = rule_text[i + len("CONSTRUCT"):j].strip()
    return rule_text[:i] + f"INSERT {{ GRAPH <{graph}> {template} }}" + rule_text[j:]


def apply_rules(db, max_rounds=5):
    inferred = G + "inferred"
    db.clear_graph(inferred)
    rules = sorted((ROOT / "rules").glob("*.rq"))
    for rnd in range(1, max_rounds + 1):
        before = db.count(inferred)
        for r in rules:
            t0 = time.time()
            db.update(construct_to_insert(r.read_text(encoding="utf-8"), inferred))
            if rnd == 1:
                print(f"    {r.stem:<28} {time.time() - t0:.2f}s")
        after = db.count(inferred)
        print(f"  round {rnd}: inferred graph {before} -> {after} triples")
        if after == before:
            break


def run_cqs(db):
    out = ROOT / "results" / "graphdb"
    out.mkdir(parents=True, exist_ok=True)
    for q in sorted((ROOT / "queries").glob("CQ*.rq")):
        try:
            csv_text = db.select(q.read_text(encoding="utf-8"), accept="text/csv")
            (out / f"{q.stem}.csv").write_text(csv_text, encoding="utf-8")
            n = max(0, len(csv_text.strip().splitlines()) - 1)
            print(f"  {q.stem:<6} {n:>4} rows")
        except RuntimeError as e:
            print(f"  {q.stem:<6} ERROR {e}")
    print(f"  CSV results in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7200")
    ap.add_argument("--repo", default="ffonto-kg")
    ap.add_argument("--config", default=str(ROOT / "graphdb" / "repo-config.ttl"))
    ap.add_argument("--recreate", action="store_true", help="delete and recreate the repository")
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="(re)load only these graphs")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--no-queries", action="store_true")
    a = ap.parse_args()

    db = GraphDB(a.url, a.repo)
    try:
        existing = db.repos()
    except Exception as e:
        sys.exit(f"GraphDB not reachable at {a.url}: {e}")
    if a.recreate and a.repo in existing:
        print(f"deleting repository {a.repo}")
        db.delete(); existing.remove(a.repo)
    if a.repo not in existing:
        print(f"creating repository {a.repo} from {a.config}")
        db.create(a.config)

    print("loading named graphs")
    for name in (a.only or SOURCES):
        files = [p for p in SOURCES[name] if p.exists()]
        if not files:
            print(f"  {name:<9} (no file – run scripts/lift.py)"); continue
        for k, p in enumerate(files):
            db.put_graph(G + name, p, replace=(k == 0))
        print(f"  {name:<9} <{G + name}>  {db.count(G + name)} triples")

    if not a.no_rules:
        print("applying rules into <%sinferred>" % G)
        apply_rules(db)
    if not a.no_queries:
        print("running CQ queries against GraphDB")
        run_cqs(db)


if __name__ == "__main__":
    main()
