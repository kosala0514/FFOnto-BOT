#!/usr/bin/env bash
# =====================================================================
#  Build the FFOnto HTML documentation with WIDOCO (ready for GitHub Pages)
#
#    bash docs/widoco/build_docs.sh <WIDOCO jar or folder> [output folder, default: site] [--oops]
#
#  e.g.  bash docs/widoco/build_docs.sh ~/WIDOCO
#        bash docs/widoco/build_docs.sh ~/WIDOCO/widoco-1.4.25-jar-with-dependencies_JDK-17.jar site --oops
#
#  Result:
#    site/index.html          core schema   (ffo:)   + WebVOWL diagram
#    site/vocab/index.html    vocabulary    (ffv:)   + concept hierarchy
#    site/rules/index.html    danger rules  (ffr:)
#    site/resources/          .ttl modules, SHACL shapes, JSON-LD context
#  Preview:  python3 -m http.server -d site 8080   ->  http://localhost:8080
#  (opening index.html directly from disk shows empty sections: WIDOCO loads them with JavaScript)
#
#  Needs: Java (17+ for the *_JDK-17.jar), and `robot` on PATH (only for the vocabulary hierarchy).
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."                       # kit root

W="${1:-}"; OUT="${2:-site}"; OOPS=""
for a in "$@"; do [ "$a" = "--oops" ] && OOPS="-oops"; done
[ "$OUT" = "--oops" ] && OUT="site"

if [ -d "$W" ]; then
  JAR=$(ls "$W"/*.jar 2>/dev/null | grep -i widoco | sort -V | tail -1 || true)
else
  JAR="$W"
fi
[ -n "${JAR:-}" ] && [ -f "$JAR" ] || { echo "WIDOCO jar not found. Usage: bash docs/widoco/build_docs.sh <WIDOCO jar or folder> [site]"; exit 1; }
command -v java >/dev/null || { echo "java not found"; exit 1; }
echo "WIDOCO: $JAR"
java -jar "$JAR" --version 2>/dev/null | grep -i version || true

ARGS=(-getOntologyMetadata -rewriteAll -lang en -excludeProvenance)
run() {                    # run <ontology file> <out folder> [extra flags]
  local ont="$1" out="$2"; shift 2
  echo; echo "=== WIDOCO: $ont -> $out"
  java -jar "$JAR" -ontFile "$ont" -outFolder "$out" "${ARGS[@]}" "$@" 2>&1 \
    | grep -E "ERROR|Exception|successfully" | grep -v "Log4j2" || true
  if [ ! -f "$out/index-en.html" ] && [ -f "$out/doc/index-en.html" ]; then   # WIDOCO sometimes writes into doc/
    cp -r "$out/doc/." "$out/" && rm -rf "$out/doc"
  fi
  [ -f "$out/index-en.html" ] || { echo "WIDOCO did not produce $out/index-en.html"; exit 1; }
}

rm -rf "$OUT"
run ontology/ffonto-core.ttl  "$OUT"        -webVowl $OOPS
run ontology/ffonto-vocab.ttl "$OUT/vocab"
run ontology/danger-rules.ttl "$OUT/rules"

mkdir -p build
TSV=""
if command -v robot >/dev/null; then
  echo; echo "=== ROBOT: concept hierarchy for the vocabulary page"
  robot query --input ontology/ffonto-vocab.ttl --query docs/widoco/vocab_tree.rq build/vocab_tree.tsv && TSV=build/vocab_tree.tsv
else
  echo "robot not on PATH – the vocabulary page will have no hierarchy (cross-reference section is still complete)"
fi

echo; echo "=== Custom sections"
PY=$(command -v python3 || command -v python)
"$PY" docs/widoco/make_sections.py --site "$OUT" ${TSV:+--vocab-tsv "$TSV"}

echo
echo "Done: $OUT/index.html"
echo "Preview:  python3 -m http.server -d $OUT 8080   then open http://localhost:8080"
