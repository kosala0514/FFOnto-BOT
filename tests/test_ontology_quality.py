"""
Structural checks on the ontology files (complements ROBOT report / OOPS!).
"""
import re, warnings
from collections import defaultdict
from pathlib import Path
from rdflib import Graph, RDF, RDFS, OWL, Namespace, URIRef

ROOT = Path(__file__).resolve().parent.parent
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
FFO = "http://purl.org/ffonto#"


def onto():
    g = Graph()
    for f in ["ffonto-core.ttl", "ffonto-vocab.ttl", "danger-rules.ttl"]:
        g.parse(ROOT / "ontology" / f, format="turtle")
    return g


def test_no_duplicate_labels():
    g = onto()
    by_label = defaultdict(set)
    for s, o in g.subject_objects(RDFS.label):
        if isinstance(s, URIRef):
            by_label[str(o)].add(s)
    dups = {k: v for k, v in by_label.items() if len(v) > 1}
    assert not dups, dups


def test_every_core_term_has_label_and_comment():
    g = onto()
    for t in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty):
        for s in g.subjects(RDF.type, t):
            if str(s).startswith(FFO):
                assert (s, RDFS.label, None) in g, f"{s} has no rdfs:label"
                assert (s, RDFS.comment, None) in g, f"{s} has no rdfs:comment"


def test_every_concept_in_a_scheme_and_connected():
    g = onto()
    for c in g.subjects(RDF.type, SKOS.Concept):
        assert (c, SKOS.inScheme, None) in g, f"{c} not in a scheme"
        assert (c, SKOS.broader, None) in g or (c, SKOS.topConceptOf, None) in g, f"{c} is an orphan"


def test_core_terms_are_used():
    """Every ffonto term should be used by a query, rule, shape, context or the lifter (CQ traceability)."""
    used_text = "\n".join(p.read_text(encoding="utf-8") for p in
                          list((ROOT / "queries").glob("*.rq")) + list((ROOT / "rules").glob("*.rq")) +
                          [ROOT / "ontology" / "ffonto-shapes.ttl", ROOT / "context" / "ffonto.context.jsonld",
                           ROOT / "scripts" / "lift.py", ROOT / "ontology" / "danger-rules.ttl",
                           ROOT / "ontology" / "ffonto-vocab.ttl"])
    g = onto()
    unused = []
    for t in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty):
        for s in g.subjects(RDF.type, t):
            if str(s).startswith(FFO):
                local = str(s)[len(FFO):]
                if not re.search(rf"ffonto:{local}\b", used_text):
                    unused.append(local)
    if unused:
        warnings.warn(f"ffonto terms not used by any CQ query/rule/mapping: {sorted(unused)}")
