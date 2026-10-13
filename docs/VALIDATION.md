# Validating FFOnto – step-by-step guide

The ontology is validated in **five layers**. Each layer catches a different kind of problem, and together they make a solid "Evaluation" section for a paper.

| Layer | Question it answers | Tool | Where |
|---|---|---|---|
| 1. Syntax | Do all files parse? | rdflib / ROBOT | steps 2–3 |
| 2. Logic | Is the ontology consistent, and is every class satisfiable? Is the data consistent with it? | **HermiT** (Protégé and ROBOT) | steps 4–6 |
| 3. Modelling quality | Are there common modelling pitfalls? | **OOPS!** | step 7 |
| 4. Engineering hygiene | Are labels, definitions, metadata and the DL profile correct? | **ROBOT report** and `validate-profile` | step 8 |
| 5. Fitness for purpose | Does the KG answer the competency questions correctly? Is the data well-formed? | **CQ test suite** (pytest) and **SHACL** | steps 9–10 |

Run every command from the repository root (`ffonto-v2/`).

---

## Step 0 – Why the ontology was changed before validation

Three things would otherwise have made HermiT or ROBOT fail:

1. **`xsd:date` is not in the OWL 2 datatype map.** HermiT stops with "unsupported datatype". Weather dates are now stored as `xsd:dateTime` (`2012-08-28T00:00:00Z`).
2. **Decimal vs double.** `ffonto:thresholdValue` has range `xsd:double`, but `30.0` in Turtle is an `xsd:decimal`. In OWL these value spaces are disjoint, so HermiT would report the whole ontology as *inconsistent*. The danger rules now write `30.0e0`.
3. **`owl:imports` of purl.org IRIs** don't resolve to your local files. Now `ontology/ffonto.ttl` is a hub that imports the three modules, and `ontology/catalog-v001.xml` maps each IRI to its local file. The SOSA, PROV-O and SKOS terms FFOnto uses are declared locally, so nothing has to be downloaded.

Definitions are now `rdfs:comment`, which OOPS! and ROBOT recognise, and every core term has one.

---

## Step 1 – Install the tools (one time)

| Tool | Install | Check |
|---|---|---|
| Java 11+ | `sudo apt install openjdk-17-jre` (Ubuntu) | `java -version` |
| ROBOT | Download `robot.jar` and the `robot` script from the latest release at <https://github.com/ontodev/robot/releases>. Put both in `~/bin`, run `chmod +x ~/bin/robot`, and add `~/bin` to your `PATH` | `robot --version` |
| Protégé 5.6.x | <https://protege.stanford.edu> (HermiT is bundled) | opens normally |
| Python packages | `pip install -r requirements.txt` | `pytest --version` |

---

## Step 2 – Build the KG once (needed for the data checks)

```bash
python scripts/lift.py        # data/kg.ttl and data/graphs/*.ttl
python scripts/run_cqs.py     # applies rules -> data/kg_inferred.ttl, results/report.md
```

---

## Step 3 – Run the whole ROBOT/HermiT pipeline in one command

```bash
bash validation/validate_ontology.sh
```

The script runs steps 4, 5, 6 and 8 below and prints PASS/FAIL for each. All outputs go to `build/`. The following sections explain each step and what to do when it fails.

---

## Step 4a – HermiT without Protégé or ROBOT (Python)

```bash
pip install owlready2            # bundles HermiT; needs Java on PATH (java -version)
python validation/reason_hermit.py
```

The script checks three things:
- the schema is consistent and has no unsatisfiable classes
- the knowledge-graph data is consistent with the schema
- all 4 negative tests are detected as inconsistent

It also writes HermiT's inferred types to `build/hermit_inferred_types.tsv`. Protégé is **optional**: use it only if you want screenshots or the interactive Explain view.

## Step 4b – HermiT in Protégé (optional, interactive, good for screenshots)

1. **Open the hub.** Choose *File → Open…* → `ontology/ffonto.ttl`. Protégé finds `catalog-v001.xml` in the same folder and loads the three modules. Check *Active ontology → Ontology imports*: all three should be listed with no red error.
2. **Pick the reasoner.** Choose *Reasoner → HermiT*.
3. **Start reasoning.** Choose *Reasoner → Start reasoner* (Ctrl+R).
4. **Read the result.**
   - If you see an *"Inconsistent ontology"* dialog, click **Explain**. Protégé shows the minimal set of axioms that clash, called a *justification*.
   - Open the *Entities → Classes* tab and switch the hierarchy to **Inferred**. If **owl:Nothing** has any subclasses, those classes are *unsatisfiable* and shown in red. Click a class, then the **?** icon, to see why.
   - **Expected result:** the ontology is consistent and owl:Nothing has no subclasses.
5. **Check the data as well.** Open `build/ffonto-with-data.owl`, which step 3 created (ontology plus `kg_inferred.ttl`), and start HermiT again.
   - Under *Individuals by class*, select an observation such as `…/obs/0`. Its yellow, inferred types include `sosa:Observation`, derived from `ffonto:ModelObservation ⊑ sosa:Observation`. Image records are likewise inferred to be `ffonto:DataRecord` and `prov:Entity`.
   - Open the *DL Query* tab and try:
     - `ModelObservation and (aboveThreshold value true)` returns the detections.
     - `SceneEntity and (wears value Helmet)` returns the helmeted people.
     - `Hazard` returns every derived hazard.
6. **Screenshot for the paper:** the inferred class hierarchy plus one Explain dialog from a negative test (step 6).

---

## Step 5 – HermiT with ROBOT (repeatable, scriptable)

```bash
# merge the modules through the catalog
robot merge --catalog ontology/catalog-v001.xml --input ontology/ffonto.ttl --output build/ffonto-merged.owl

# schema: consistency + unsatisfiable classes
robot reason --reasoner HermiT --input build/ffonto-merged.owl \
      --equivalent-classes-allowed asserted-only --exclude-tautologies structural \
      --dump-unsatisfiable build/unsatisfiable.owl --output build/ffonto-reasoned.owl

# data: is the knowledge graph consistent with the schema?
robot merge --input build/ffonto-merged.owl --input data/kg_inferred.ttl --output build/ffonto-with-data.owl
robot reason --reasoner HermiT --input build/ffonto-with-data.owl --output build/ffonto-with-data-reasoned.owl
```

If a `reason` command fails, ask ROBOT for the justification:

```bash
robot explain --input build/ffonto-with-data.owl --reasoner HermiT -M inconsistency --explanation build/explain-data.md
```

Common causes and fixes:

| Message in the explanation | Cause | Fix |
|---|---|---|
| `DataPropertyRange(... xsd:double)` with a `"…"^^xsd:decimal` or `xsd:integer` value | Wrong literal type in the data | Cast with `float()` in `lift.py`, or write `1.0e0` in Turtle |
| `DisjointClasses(...)` | One individual got two disjoint types, often through a domain/range axiom | Check the property's domain/range. Is the data or the axiom wrong? |
| `UnsupportedDatatypeException` | A datatype outside OWL 2 (e.g. `xsd:date`, `xsd:gYear`) | Use `xsd:dateTime`, or a string |

---

## Step 6 – Negative tests (prove the axioms actually work)

A consistent ontology might simply have no constraints. Negative tests show that the reasoner **catches** wrong data. Each file in `validation/owl-negative/` breaks one axiom on purpose:

| File | Wrong data | Axiom that must catch it |
|---|---|---|
| `neg01_disjoint_records.ttl` | Something is both ImageRecord and AudioRecord | `owl:disjointWith` |
| `neg02_bad_datatype.ttl` | `ffonto:confidence "high"` | range `xsd:double` |
| `neg03_disjoint_entity_kinds.ttl` | Something is both Vehicle and BuiltAsset | `owl:AllDisjointClasses` |
| `neg04_wears_on_building.ttl` | A building "wears" a helmet | domain of `ffonto:wears` plus disjointness |

`validate_ontology.sh` reports **PASS** when HermiT finds each file inconsistent. To test one by hand:

```bash
robot merge --input build/ffonto-merged.owl --input validation/owl-negative/neg01_disjoint_records.ttl --output build/neg.owl
robot reason --reasoner HermiT --input build/neg.owl        # expected: "ontology is inconsistent"
```

When you add a new constraint to the ontology, add a negative test for it as well.

---

## Step 7 – OOPS! (pitfall scanner)

OOPS! checks about 40 common modelling mistakes and grades them *Critical*, *Important* or *Minor*. It only accepts **RDF/XML**.

**Option A – script (uses the OOPS! REST service):**

```bash
python validation/oops_check.py          # core schema (recommended)
python validation/oops_check.py --all    # + vocabulary and danger rules
```

This writes `build/oops_report.md` (a summary table) and `build/oops_response.xml` (the raw response).

**Option B – web form:**

1. Run `robot convert --input ontology/ffonto-core.ttl --output build/ffonto-core.owl`, or use `build/oops_input.owl` created by the script.
2. Open <https://oops.linkeddata.es> → **"Scanner by direct input"**. Paste the RDF/XML and click **Go!**
3. Save the results page as a PDF for your appendix.

**How to read the results:**

| Grade | Action |
|---|---|
| Critical | Must fix (e.g. P05 wrong inverse, P19 multiple domains/ranges, P28/P29 wrong symmetric/transitive use) |
| Important | Fix, or justify in the paper |
| Minor | Usually naming or annotation issues. Fix the cheap ones |

Pitfalls you can expect and **justify by design**:

- **P13** (inverse relationships not declared): FFOnto deliberately keeps one direction per relation, and SPARQL uses `^path` for the other.
- **P11** (missing domain or range): a few properties are intentionally open, e.g. `ffonto:threatens` points to entities *or* concepts. ROBOT lists them at INFO level (step 8).

Fix everything else, then re-run until only justified pitfalls remain. For the paper, report a before/after table: the old FFOnto versus v2.

---

## Step 8 – ROBOT report and DL profile

```bash
robot validate-profile --profile DL --input build/ffonto-merged.owl --output build/dl-profile.txt
robot report --input build/ffonto-merged.owl --profile validation/robot/profile.txt \
      --fail-on ERROR --print 20 --output build/robot-report.tsv
robot report --input build/ffonto-merged.owl --profile validation/robot/profile.txt \
      --fail-on none --output build/robot-report.html          # readable version
robot measure --input build/ffonto-merged.owl --metrics extended --output build/metrics.tsv
```

- **OWL 2 DL profile:** HermiT is only guaranteed complete for OWL 2 DL. Expected output: *no violations*.
- **`validation/robot/profile.txt`** uses ROBOT's standard checks except the OBO-specific ones. ROBOT's `missing_definition` looks for the OBO definition property (`IAO:0000115`), so it is replaced by FFOnto checks in `validation/robot/queries/`:
  - `missing_comment` (WARN): every `ffonto:` class/property has an `rdfs:comment`
  - `concept_missing_preflabel` (ERROR) and `concept_missing_scheme` (ERROR): SKOS hygiene
  - `orphan_concept` (ERROR): every concept has `skos:broader` or is a top concept
  - `property_missing_domain_or_range` (INFO): open properties are listed for review
- **Expected output:** no ERROR rows. Typical INFO rows are `missing_superclass` (top-level classes) and the open properties.
- **`metrics.tsv`** gives class, property, individual and axiom counts plus DL expressivity. Put these in the paper's ontology-description table.

---

## Step 9 – CQ test suite (does the KG answer the questions?)

```bash
pytest -q tests/                       # everything
pytest -q tests/test_cq_suite.py -k CQ17   # one competency question
```

What the tests check:

| Test file | Checks |
|---|---|
| `tests/test_cq_suite.py` | Every CQ01–CQ26 has a query, the query runs, and its result contains the expected rows from `tests/cq_expectations.py` |
| `tests/test_data_quality.py` | SHACL finds no `sh:Violation` in the KG (warnings allowed) |
| `tests/test_ontology_quality.py` | No duplicate labels; every core term has a label and comment; every concept is in a scheme and connected; unused `ffonto:` terms are reported (CQ traceability) |

**Using the VQA dataset as ground truth.** This is the most valuable step:

1. Pick 30–50 images that have VQA answers. Run the VLM on them (`prompts/scene_extraction_prompt.md`) and run `lift.py`.
2. For each image's questions, add the ground-truth answer as an expectation:
   ```python
   "CQ17": {"rows": [{"record": "~<uuid of forest_fire_081>", "missingRequiredPPE": ""}]},  # VQA: "they are wearing helmets, suits, gloves"
   ```
3. Run `pytest` and count pass/fail per CQ. **Accuracy per CQ = passed / total.** Failures fall into three kinds:
   - the ontology is missing a concept → fix the ontology
   - the VLM extracted the wrong facts → a perception error, not an ontology error
   - the rule or query is wrong → fix the query

   Report these three error sources separately in the paper.

---

## Step 10 – SHACL report (data quality)

`python scripts/run_cqs.py` already writes `results/shacl_report.txt`. Expected warnings with the sample data:

- the real image and audio have no `capture_time_utc` or `crs`
- the 2012-08-27 weather row has no `event_id`
- the caption model has no version

These describe your real data's gaps, so list them as data-quality findings.

---

## Checklist

- [ ] `validate_ontology.sh` → ALL VALIDATION STEPS PASSED
- [ ] Protégé + HermiT: consistent, owl:Nothing has no subclasses; screenshots saved
- [ ] Negative tests: 4/4 detected
- [ ] OOPS!: no Critical; Important ones fixed or justified; report saved
- [ ] ROBOT report: 0 ERROR; `metrics.tsv` saved
- [ ] `pytest tests/` green; per-CQ accuracy table built from VQA ground truth
- [ ] SHACL warnings listed as data-quality findings
