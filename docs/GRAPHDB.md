# Storing the FFOnto KG in GraphDB

## Decision: one repository with named graphs (not one repository per modality)

**Recommendation: create one repository (`ffonto`) and put each modality in its own named graph.**

| | One repo + named graphs (recommended) | One repo per modality (image / audio / tabular) |
|---|---|---|
| Cross-modal questions (CQ01, CQ07, CQ24 – "do image and audio of event X agree?") | A normal SPARQL join | Needs federation (`SERVICE`) across repos: slower and more complex queries |
| Reasoning (RDFS-Plus) | Schema and data are in one place, so inference works across all data | The schema must be copied into every repo; there is no inference across repos |
| Rules R00–R07 | Run once over everything | Rules that join modalities (conflicts, hazards) cannot run |
| Reloading one modality | `PUT` replaces just that named graph | Reload that repo |
| Provenance ("where did this triple come from?") | `GRAPH ?g { … }` tells you | Implicit, by repo |
| Querying a single modality | `FROM <…/graph/audio>` | Query that repo |
| Operations | One endpoint, one backup, one config | N endpoints, N backups, N configs |

The point of the project is **multimodal fusion**, which means joining image, audio and weather on `ffonto:aboutEvent`. Separate repositories break exactly that.

Named graphs give you the separation benefits anyway: provenance, per-modality reload and per-modality queries.

**When separate repositories do make sense:** use them for *lifecycle*, not *modality*. For example, `ffonto-dev` and `ffonto-prod`, or a repo per experiment or dataset version. Another case is a modality so large it needs its own server, such as millions of frame-level observations. You can still federate those later.

### Named-graph layout

| Named graph | Content | Source file |
|---|---|---|
| `http://purl.org/ffonto/graph/ontology` | core schema + danger rules | `ontology/ffonto-core.ttl`, `ontology/danger-rules.ttl` |
| `http://purl.org/ffonto/graph/vocab` | SKOS vocabulary + background knowledge | `ontology/ffonto-vocab.ttl` |
| `http://purl.org/ffonto/graph/image` | image records, classifier observations, captions | `data/graphs/image.ttl` |
| `http://purl.org/ffonto/graph/audio` | audio records, sound events | `data/graphs/audio.ttl` |
| `http://purl.org/ffonto/graph/weather` | weather / FWI rows | `data/graphs/weather.ttl` |
| `http://purl.org/ffonto/graph/scene` | VLM scene descriptions | `data/graphs/scene.ttl` |
| `http://purl.org/ffonto/graph/inferred` | output of rules R00–R07 (rebuilt on every load) | generated |

In GraphDB, a query without `FROM` sees the **union of all graphs**. That means the 27 CQ queries work unchanged.

---

## Step 1 – Install GraphDB

Pick one option.

- **Docker:**
  ```bash
  docker run -d --name graphdb -p 7200:7200 -v graphdb-home:/opt/graphdb/home ontotext/graphdb:<version>
  ```
  Use a current 10.x or 11.x tag from Docker Hub (`ontotext/graphdb`). The free edition needs no licence file.
- **Desktop:** download GraphDB from <https://graphdb.ontotext.com> and start it.

Open the Workbench at <http://localhost:7200>.

## Step 2 – Create the repository

**Option A – Workbench UI:**

1. *Setup → Repositories → Create new repository → GraphDB Repository*.
2. Set **Repository ID** to `ffonto`.
3. Set **Ruleset** to **RDFS-Plus (Optimized)**. This gives subClassOf, subPropertyOf, domain/range, and symmetric `ffonto:near`. The ruleset cannot be changed later.
4. Tick **Disable owl:sameAs**.
5. Tick **Use context index**. This makes queries filtered by named graph fast.
6. Click **Create**, then select `ffonto` in the repository drop-down (top right).

**Option B – config file:**

```bash
curl -X POST http://localhost:7200/rest/repositories -F "config=@graphdb/repo-config.ttl"
```

Why RDFS-Plus and not OWL2-RL? Full OWL consistency checking is already done offline with HermiT (VALIDATION.md). Inside the store you only need cheap, predictable inference. Heavier rulesets slow down loading and can add surprising triples.

## Step 3 – Produce the per-modality files

```bash
python scripts/lift.py          # writes data/graphs/image.ttl, audio.ttl, weather.ttl, scene.ttl
```

## Step 4 – Load everything, apply the rules and run the CQs (one command)

```bash
pip install requests
python graphdb/load_graphdb.py                    # creates the repo if missing
python graphdb/load_graphdb.py --recreate         # start from scratch
python graphdb/load_graphdb.py --only image scene # reload just these graphs (rules are re-applied)
```

The script does four things:

1. Uploads each file into its named graph with the RDF4J Graph Store protocol. `PUT …/rdf-graphs/service?graph=<iri>` *replaces* that graph only.
2. Rewrites each rule's `CONSTRUCT` into `INSERT { GRAPH <…/inferred> { … } } WHERE { … }` and runs it as a SPARQL Update. It repeats until nothing new is derived.
3. Runs `queries/CQ01–CQ26.rq` against the endpoint.
4. Saves `results/graphdb/CQxx.csv`. The CSVs should match `results/*.csv` from the offline run.

**Manual alternative (Workbench):** *Import → Upload RDF files*. Select a file, choose **Target graphs → Named graph**, enter the IRI from the table above, then click Import. Repeat for each file. Then paste each rule from `rules/` into *SPARQL*, changing `CONSTRUCT {…}` to `INSERT { GRAPH <http://purl.org/ffonto/graph/inferred> {…} }`.

## Step 5 – Check the load

In *SPARQL*:

```sparql
# triples per named graph
SELECT ?g (COUNT(*) AS ?triples) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g ORDER BY ?g
```

```sparql
# cross-modal check: which events have more than one modality?
PREFIX ffonto: <http://purl.org/ffonto#>
SELECT ?event (GROUP_CONCAT(DISTINCT STRAFTER(STR(?g), "graph/"); separator=", ") AS ?modalities)
WHERE { GRAPH ?g { ?rec ffonto:aboutEvent ?event } }
GROUP BY ?event HAVING (COUNT(DISTINCT ?g) > 1)
```

```sparql
# query one modality only
PREFIX ffonto: <http://purl.org/ffonto#>
SELECT ?obs ?class ?conf
FROM <http://purl.org/ffonto/graph/audio>
WHERE { ?obs ffonto:predictedClass ?class ; ffonto:confidence ?conf }
```

To see only explicit triples, without GraphDB's RDFS inference, add `FROM <http://www.ontotext.com/explicit>`.

## Step 6 – Explore and present

- **Visual graph** (*Explore → Visual graph*): start from an event IRI such as `http://purl.org/ffonto/data/event/ev-demo-081`. You will see its image, audio and weather records, their observations and the derived hazards. This makes a good figure for the paper.
- **Saved queries:** save CQ01–CQ26 in the Workbench, so supervisors can run them without Python.
- **Class hierarchy** (*Explore → Class hierarchy*): a quick visual check of the schema.

## Step 7 – Day-to-day workflow

| Change | Command |
|---|---|
| New images or VLM scene files | `python scripts/lift.py` then `python graphdb/load_graphdb.py --only image scene` |
| New audio clips | `python scripts/lift.py` then `python graphdb/load_graphdb.py --only audio` |
| Edited ontology or vocabulary | Validate first (VALIDATION.md), then `python graphdb/load_graphdb.py --only ontology vocab` |
| Changed rules or thresholds | `python graphdb/load_graphdb.py --only ontology` (rules re-run automatically) |
| Backup | Workbench *Repositories → ⋮ → Download as TriG/N-Quads*. This keeps the named graphs |

## Notes and caveats

- **SHACL:** keep validating with `pyshacl` in the Python pipeline (`run_cqs.py`, `pytest`). GraphDB has built-in SHACL validation, but its engine may not support SHACL-SPARQL constraints (two FFOnto shapes use `sh:sparql`). If you enable it, load only the core shapes into GraphDB's SHACL shape graph.
- **Inference side-effects:** with RDFS-Plus, `ffonto:near` is symmetric, and sub-property triples are inferred (e.g. `ffonto:spatiallyRelated`). The CQ queries already handle this (CQ21 filters the super-property).
- **Scale:** the context index plus one named graph per modality scales to tens of millions of triples on a normal workstation. For very large frame-level audio or video data, consider aggregating per clip before loading.
