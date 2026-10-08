"""SHACL validation of the knowledge graph must produce no sh:Violation (warnings are allowed)."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_shacl_no_violations(kg):
    pyshacl = pytest.importorskip("pyshacl")
    from rdflib import Graph
    shapes = Graph().parse(ROOT / "ontology" / "ffonto-shapes.ttl", format="turtle")
    conforms, report_graph, text = pyshacl.validate(kg, shacl_graph=shapes, inference="none",
                                                     advanced=True, allow_warnings=True)
    assert conforms, text
