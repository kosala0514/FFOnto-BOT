"""
The CQ test suite runs on its OWN fixed test data in tests/fixtures/ – not on data/raw.
That way the expected answers in cq_expectations.py stay valid no matter what real
data you put in data/raw (real data is evaluated separately against VQA ground truth).
"""
import importlib.util, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
BUILD = ROOT / "tests" / "_build"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_cqs", ROOT / "scripts" / "run_cqs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def runner():
    return _load_runner()


@pytest.fixture(scope="session")
def kg(runner):
    """Lift the FIXTURE data, load ontology + data, apply all rules (once per test session)."""
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "lift.py"),
                    "--raw", str(FIX / "raw"), "--scenes", str(FIX / "scenes"), "--out", str(BUILD)],
                   check=True, cwd=ROOT)
    g = runner.load_graph(BUILD / "kg.ttl")
    runner.apply_rules(g, out_file=BUILD / "kg_inferred.ttl")
    return g
