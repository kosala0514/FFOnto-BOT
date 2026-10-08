#!/usr/bin/env bash
# =====================================================================
#  FFOnto – ontology validation pipeline (ROBOT + HermiT)
#  Run from the repository root:   bash validation/validate_ontology.sh
#  Needs: Java 11+ and the `robot` command (https://robot.obolibrary.org)
#  Data step needs data/kg_inferred.ttl  ->  run scripts/run_cqs.py first.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
CAT=ontology/catalog-v001.xml
FAILED=0
step() { echo; echo "=== $1"; }
ok()   { echo "    PASS: $1"; }
bad()  { echo "    FAIL: $1"; FAILED=1; }

command -v robot >/dev/null || { echo "robot not found – see docs/VALIDATION.md step 1"; exit 1; }

step "1. Merge modules (core + vocabulary + danger rules) via the import catalog"
if robot merge --catalog "$CAT" --input ontology/ffonto.ttl --output build/ffonto-merged.owl; then
  ok "build/ffonto-merged.owl"
else
  bad "merge failed (syntax error or unresolved import)"; exit 1
fi

step "2. OWL 2 DL profile check"
if robot validate-profile --profile DL --input build/ffonto-merged.owl --output build/dl-profile.txt; then
  ok "ontology is in OWL 2 DL (HermiT results are complete)"
else
  bad "OWL 2 DL violations – see build/dl-profile.txt"
fi

step "3. HermiT: consistency + unsatisfiable classes (schema + vocabulary)"
if robot reason --reasoner HermiT --input build/ffonto-merged.owl \
      --equivalent-classes-allowed asserted-only --exclude-tautologies structural \
      --dump-unsatisfiable build/unsatisfiable.owl \
      --output build/ffonto-reasoned.owl; then
  ok "consistent, no unsatisfiable classes -> build/ffonto-reasoned.owl"
else
  bad "reasoning failed – explaining…"
  robot explain --input build/ffonto-merged.owl --reasoner HermiT -M inconsistency \
        --explanation build/explain-schema.md || true
  echo "    see build/explain-schema.md and build/unsatisfiable.owl"
fi

step "4. HermiT: knowledge-graph data against the schema (ABox consistency)"
if [ -f data/kg_inferred.ttl ]; then
  robot merge --input build/ffonto-merged.owl --input data/kg_inferred.ttl --output build/ffonto-with-data.owl
  if robot reason --reasoner HermiT --input build/ffonto-with-data.owl --output build/ffonto-with-data-reasoned.owl; then
    ok "data is consistent with the ontology"
  else
    bad "data makes the ontology inconsistent – explaining…"
    robot explain --input build/ffonto-with-data.owl --reasoner HermiT -M inconsistency \
          --explanation build/explain-data.md || true
    echo "    see build/explain-data.md"
  fi
else
  echo "    SKIP: data/kg_inferred.ttl not found (run python scripts/run_cqs.py first)"
fi

step "5. Negative tests – each file MUST be detected as inconsistent"
for t in validation/owl-negative/*.ttl; do
  robot merge --input build/ffonto-merged.owl --input "$t" --output build/neg.owl >/dev/null 2>&1
  if robot reason --reasoner HermiT --input build/neg.owl >/dev/null 2>&1; then
    bad "$(basename "$t") was NOT detected (axiom missing?)"
  else
    ok "$(basename "$t") correctly detected as inconsistent"
  fi
done
rm -f build/neg.owl

step "6. ROBOT report (labels, metadata, SKOS hygiene, definitions)"
robot report --input build/ffonto-merged.owl --profile validation/robot/profile.txt \
      --fail-on none --output build/robot-report.html >/dev/null
if robot report --input build/ffonto-merged.owl --profile validation/robot/profile.txt \
      --fail-on ERROR --print 20 --output build/robot-report.tsv; then
  ok "no ERROR-level problems (warnings/info listed in build/robot-report.tsv/.html)"
else
  bad "ERROR-level problems – see build/robot-report.tsv"
fi

step "7. Metrics for the paper"
robot measure --input build/ffonto-merged.owl --metrics extended --output build/metrics.tsv && ok "build/metrics.tsv"

step "8. RDF/XML export for OOPS! (core schema only)"
robot convert --input ontology/ffonto-core.ttl --output build/ffonto-core.owl && ok "build/ffonto-core.owl"

echo
if [ "$FAILED" = 0 ]; then echo "ALL VALIDATION STEPS PASSED"; else echo "SOME STEPS FAILED – see messages above"; fi
exit $FAILED
