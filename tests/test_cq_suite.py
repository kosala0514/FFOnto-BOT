"""
CQ test suite: every competency question must (1) have a query, (2) run without error,
and (3) return the expected answers on the test data.

    pytest -q tests/test_cq_suite.py            # all
    pytest -q tests/test_cq_suite.py -k CQ17    # one CQ
"""
from pathlib import Path
import pytest
from cq_expectations import EXPECT

ROOT = Path(__file__).resolve().parent.parent
QUERY_DIR = ROOT / "queries"
ALL_CQS = [f"CQ{i:02d}" for i in range(1, 27)]


def run(runner, g, cq):
    res = g.query((QUERY_DIR / f"{cq}.rq").read_text(encoding="utf-8"))
    cols = [str(v) for v in res.vars]
    return [{c: runner.fmt(g, r[c]) for c in cols} for r in res]


def matches(row, pattern):
    for col, exp in pattern.items():
        val = row.get(col, "")
        if exp.startswith("~"):
            if exp[1:] not in val:
                return False
        elif val != exp:
            return False
    return True


@pytest.mark.parametrize("cq", ALL_CQS)
def test_query_exists(cq):
    assert (QUERY_DIR / f"{cq}.rq").exists(), f"{cq} has no SPARQL query"


@pytest.mark.parametrize("cq", sorted(EXPECT))
def test_cq_answers(runner, kg, cq):
    spec = EXPECT[cq]
    rows = run(runner, kg, cq)
    assert len(rows) >= spec.get("min_rows", 1), f"{cq} returned {len(rows)} rows"
    for pat in spec.get("rows", []):
        assert any(matches(r, pat) for r in rows), (
            f"{cq}: no row matches {pat}\nfirst rows: {rows[:5]}")
    for pat in spec.get("absent", []):
        assert not any(matches(r, pat) for r in rows), f"{cq}: unexpected row matching {pat}"


def test_every_cq_has_expectation():
    missing = [cq for cq in ALL_CQS if cq not in EXPECT]
    assert not missing, f"CQs without expected answers: {missing}"
