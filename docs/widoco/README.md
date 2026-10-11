# Publishing the FFOnto documentation (WIDOCO → GitHub Pages)

The result is three linked pages with a shared download folder:

| Page | Documents | Extras |
|---|---|---|
| `site/index.html` | core schema, `ffonto:` (classes and properties) | 3-layer diagram, design principles, CQ table generated from `queries/*.rq`, example, WebVOWL diagram |
| `site/vocab/index.html` | vocabulary, `ffontov:` (374 SKOS concepts) | collapsible concept hierarchy per scheme, keywords |
| `site/rules/index.html` | danger rules, `ffontor:` | — |
| `site/resources/` | `.ttl` modules, SHACL shapes, JSON-LD context | downloads |

The introduction, description and references sections are written in `docs/widoco/sections/*.html`. WIDOCO
generates everything else from the ontology files: metadata, overview, cross-reference and serializations.
The title, abstract, authors, version, licence and citation come from the ontology header, so edit them **there**
(`ontology/ffonto-core.ttl`). The FOOPS! FAIRness check reads the same header.

## 1. Build

Requirements: Java (17 or newer for the `*_JDK-17.jar`), the WIDOCO jar, and `robot` on PATH (already installed
for validation; it is used only for the vocabulary hierarchy).

```bash
cd /home/cse/Documents/Multimodal_Ontology-based_KG
bash docs/widoco/build_docs.sh ~/WIDOCO            # path to your WIDOCO folder or to the .jar itself
# optional: add --oops to also embed an OOPS! evaluation page (needs internet)
```

## 2. Preview locally

```bash
python3 -m http.server -d site 8080       # then open http://localhost:8080
```

Opening `site/index.html` directly from the file manager shows empty sections. This is normal: WIDOCO loads its
sections with JavaScript, which browsers block for `file://` pages. Check these:

- **Introduction:** the module table links to the vocabulary and rules pages.
- **Description:** Figure 1, the CQ table and the example.
- **Overview:** the WebVOWL diagram.
- **Vocabulary page:** clicking a concept in the hierarchy jumps to its entry.

## 3. Publish on GitHub Pages (your existing repository `kosala0514/ffonto_html`)

This keeps the old (v1) documentation at `/v1/` and puts v2 at the root:

```bash
cd ~
git clone https://github.com/kosala0514/ffonto_html.git
cd ffonto_html
mkdir -p v1
for f in $(ls -A | grep -v -x -e .git -e v1); do git mv "$f" v1/; done      # move old docs into v1/
cp -r /home/cse/Documents/Multimodal_Ontology-based_KG/site/. .
git add -A
git commit -m "FFOnto 2.0.0 documentation (WIDOCO)"
git push
```

Then, on GitHub, open **Settings → Pages → Build and deployment**: *Deploy from a branch*, branch `main`, folder `/ (root)`.
After a minute the site is at `https://kosala0514.github.io/ffonto_html/` (vocabulary at `…/vocab/`).
`build_docs.sh` adds the `.nojekyll` file that Pages needs to serve WIDOCO's folders unchanged.

To update later: re-run step 1, copy `site/.` into the repository again, then commit and push.

## 4. Make the ontology IRIs resolvable (recommended before the paper)

The IRIs are currently `http://purl.org/ffonto#…`. Until something redirects them, clicking `ffonto:FireEvent` leads
nowhere, and FOOPS! reports that the IRIs do not resolve. There are two ways to fix this:

| Option | What to do | Content negotiation (HTML for people, Turtle for machines) |
|---|---|---|
| **A. Keep `purl.org`** | Log in at purl.archive.org (Internet Archive), create the domain `/ffonto` and add a *partial* redirect to `https://kosala0514.github.io/ffonto_html/` | No (HTML only) |
| **B. Move to `w3id.org`** (common for ontologies) | Pull request to github.com/perma-id/w3id.org adding `ffonto/.htaccess` (template: `docs/widoco/w3id/.htaccess`) | Yes |

Option B also changes the namespace from `http://purl.org/ffonto` to `https://w3id.org/ffonto`. Do this once,
before publishing results:

```bash
grep -rl "http://purl.org/ffonto" --include=*.ttl --include=*.rq --include=*.py --include=*.jsonld --include=*.xml --include=*.md . \
  | xargs sed -i 's#http://purl.org/ffonto#https://w3id.org/ffonto#g'
bash validation/validate_ontology.sh && pytest -q tests/ && python scripts/lift.py && python graphdb/load_graphdb.py
```

Re-run the validation and tests, rebuild the KG and reload GraphDB: the stored data contains the old IRIs.
Then rebuild and republish the documentation.

## 5. After publishing

- **FOOPS!** (FAIRness of the ontology): https://foops.linkeddata.es/FAIR_validator.html. Enter the ontology IRI once it resolves, or upload `ontology/ffonto-core.ttl` before that. Report its score with the other validation results.
- Put the documentation URL in the paper, in the GitHub repository README, and in the Kaggle dataset description.
