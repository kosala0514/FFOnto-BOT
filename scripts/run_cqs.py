#!/usr/bin/env python3
"""
run_cqs.py — load ontology + vocabulary + data, apply rules, validate with SHACL,
and answer every competency question (queries/CQxx.rq).

Usage (from the repository root, after `python scripts/lift.py`):
    python scripts/run_cqs.py               # all CQs
    python scripts/run_cqs.py CQ05 CQ24     # selected CQs
    python scripts/run_cqs.py --timeout 30  # stop any single query after 30 s (default 120)
Outputs:
    results/CQxx.csv        one CSV per query
    results/report.md       all answers in one readable file
    results/shacl_report.txt
    data/kg_inferred.ttl    data + everything the rules derived
"""
import csv, signal, sys, time, traceback
from pathlib import Path

try:
    from rdflib import Graph, Literal, URIRef
except ImportError:
    sys.exit("Please install rdflib first:  pip install rdflib pyshacl")

ROOT = Path(__file__).resolve().parent.parent
ONTO = ["ontology/ffonto-core.ttl", "ontology/ffonto-vocab.ttl", "ontology/danger-rules.ttl"]
SHAPES = ROOT / "ontology" / "ffonto-shapes.ttl"
KG = ROOT / "data" / "kg.ttl"
OUT = ROOT / "results"


def load_graph(kg_path=None):
    kg_file = Path(kg_path) if kg_path else KG
    if not kg_file.exists():
        sys.exit(f"{kg_file} not found – run  python scripts/lift.py  first.")
    g = Graph()
    for f in ONTO:
        g.parse(ROOT / f, format="turtle")
    n_schema = len(g)
    g.parse(kg_file, format="turtle")
    for p, ns in [("ffonto", "http://purl.org/ffonto#"), ("ffontov", "http://purl.org/ffonto/vocab#"),
                  ("ffd", "http://purl.org/ffonto/data/"), ("sosa", "http://www.w3.org/ns/sosa/"),
                  ("prov", "http://www.w3.org/ns/prov#"), ("skos", "http://www.w3.org/2004/02/skos/core#")]:
        g.bind(p, ns, override=True)
    print(f"loaded {n_schema} schema/vocabulary triples + {len(g) - n_schema} data triples")
    return g


def apply_rules(g, max_rounds=5, out_file="default"):
    rules = sorted((ROOT / "rules").glob("*.rq"))
    for rnd in range(1, max_rounds + 1):
        before = len(g)
        for r in rules:
            t0 = time.time()
            res = g.query(r.read_text(encoding="utf-8"))
            added = 0
            for triple in res:
                if triple not in g:
                    g.add(triple)
                    added += 1
            if rnd == 1 or added:
                print(f"  round {rnd} {r.stem:<28} +{added} triples ({time.time() - t0:.2f}s)")
        if len(g) == before:
            break
    if out_file == "default":
        out_file = ROOT / "data" / "kg_inferred.ttl"
    if out_file:
        g.serialize(out_file, format="turtle")


def run_shacl(g):
    try:
        from pyshacl import validate
    except ImportError:
        print("pyshacl not installed – skipping SHACL validation (pip install pyshacl)")
        return
    shapes = Graph().parse(SHAPES, format="turtle")
    conforms, _, text = validate(g, shacl_graph=shapes, inference="none", advanced=True, allow_warnings=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "shacl_report.txt").write_text(text, encoding="utf-8")
    n = text.count("Constraint Violation")
    print(f"SHACL: conforms={conforms}; {n} results – see results/shacl_report.txt")


def fmt(g, v):
    if v is None:
        return ""
    if isinstance(v, URIRef):
        try:
            return v.n3(g.namespace_manager)
        except Exception:
            return str(v)
    if isinstance(v, Literal):
        val = v.toPython()
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)
    return str(v)


class QueryTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise QueryTimeout()


def run_queries(g, only, timeout=120):
    """Run each CQ. Errors and slow queries are reported per CQ and the loop continues.

    Note: rdflib evaluates SELECT results lazily – the query really runs while the rows are
    read – so evaluation must happen inside the try block, otherwise one failing query stops
    the whole run."""
    OUT.mkdir(exist_ok=True)
    md = ["# FFOnto v2 – competency question results\n"]
    status = []
    use_alarm = hasattr(signal, "SIGALRM") and timeout > 0
    for q in sorted((ROOT / "queries").glob("CQ*.rq")):
        if only and q.stem not in only:
            continue
        text = q.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        print(f"\n>> running {q.stem} …", flush=True)
        t0 = time.time()
        try:
            if use_alarm:
                signal.signal(signal.SIGALRM, _alarm)
                signal.alarm(timeout)
            res = g.query(text)
            cols = [str(v) for v in res.vars]
            rows = [[fmt(g, r[c]) for c in cols] for r in res]      # evaluation happens here
        except QueryTimeout:
            msg = f"TIMEOUT after {timeout}s"
            print(f"!! {q.stem} {msg}", flush=True)
            md.append(f"## {title}\n\n**{msg}**\n"); status.append((q.stem, msg))
            continue
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"!! {q.stem} failed: {msg}", flush=True)
            traceback.print_exc()
            md.append(f"## {title}\n\n**ERROR:** `{msg}`\n"); status.append((q.stem, "ERROR " + msg))
            continue
        finally:
            if use_alarm:
                signal.alarm(0)
        status.append((q.stem, f"{len(rows)} rows, {time.time() - t0:.1f}s"))
        with (OUT / f"{q.stem}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(rows)
        print(f"=== {title}  ({len(rows)} rows, {time.time() - t0:.1f}s)")
        widths = [min(40, max([len(c)] + [len(r[i]) for r in rows])) for i, c in enumerate(cols)]
        print("  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)))
        for r in rows[:25]:
            print("  " + " | ".join(r[i][:40].ljust(widths[i]) for i in range(len(cols))))
        md.append(f"## {title}\n")
        if rows:
            md.append("| " + " | ".join(cols) + " |")
            md.append("|" + "---|" * len(cols))
            md += ["| " + " | ".join(x.replace("|", "\\|") for x in r) + " |" for r in rows]
        else:
            md.append("_No rows (for yes/no questions this means **no**, or that the data needed is missing)._")
        md.append("")
        (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")   # save progress after every CQ
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\nSUMMARY")
    for cq, st in status:
        print(f"  {cq:<6} {st}")
    print(f"\nwrote results/report.md and one CSV per CQ in {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:]
    timeout = 120
    if "--timeout" in args:
        i = args.index("--timeout"); timeout = int(args[i + 1]); del args[i:i + 2]
    only = set(args)
    g = load_graph()
    print("applying rules …")
    apply_rules(g)
    run_shacl(g)
    run_queries(g, only, timeout)
