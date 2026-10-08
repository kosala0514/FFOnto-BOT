"""
Fill the custom WIDOCO sections (introduction, description, references) and install them into the generated docs.
Standard library only. Called by build_docs.sh after WIDOCO has run:

    python docs/widoco/make_sections.py --site site --vocab-tsv build/vocab_tree.tsv

What it does
  site/sections/{introduction,description,references}-en.html      core schema (CQ table + statistics)
  site/vocab/sections/{introduction,description}-en.html           vocabulary (concept hierarchy)
  site/rules/sections/introduction-en.html                          danger rules
  site/**/index.html  (copy of index-en.html: GitHub Pages serves index.html by default)
  site/resources/     the module files, SHACL shapes and JSON-LD context for download
"""
import argparse
import csv
import html
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXTRA_CSS = """
<style>
  .ffo-table{border-collapse:collapse;margin:.6em 0 1em}
  .ffo-table th,.ffo-table td{border:1px solid #ccc;padding:4px 8px;vertical-align:top;text-align:left}
  .ffo-table th{background:rgba(127,127,127,.12)}
  .ffo-figure{margin:1em 0} .ffo-figure figcaption{font-size:.9em;color:#666}
  pre.ffo-code{background:rgba(127,127,127,.1);padding:10px;overflow-x:auto;font-size:.88em}
  .ffo-tree ul{list-style:none;margin:0;padding-left:1.2em;border-left:1px dotted #bbb}
  .ffo-tree>ul{border-left:none;padding-left:0}
  .ffo-tree li{margin:2px 0} .ffo-kw{color:#888;font-size:.85em} .ffo-def{color:#555;font-size:.9em}
  details.ffo-scheme{margin:.4em 0} details.ffo-scheme summary{cursor:pointer;font-weight:bold}
</style>
"""


# ------------------------------------------------------------------ inputs
def ttl_stats():
    core = (ROOT / "ontology/ffonto-core.ttl").read_text(encoding="utf-8")
    vocab = (ROOT / "ontology/ffonto-vocab.ttl").read_text(encoding="utf-8")
    count = lambda t: len(re.findall(rf"^ffo:\w+\s+a\s+owl:{t}\b", core, re.M))
    version = re.search(r'owl:versionInfo\s+"([^"]+)"', core)
    return {
        "N_CLASSES": count("Class"), "N_OBJ": count("ObjectProperty"), "N_DATA": count("DatatypeProperty"),
        "N_CONCEPTS": len(re.findall(r"^ffv:\w+\s+a\s+skos:Concept\b", vocab, re.M)),
        "N_SCHEMES": len(re.findall(r"^ffv:\w+\s+a\s+skos:ConceptScheme\b", vocab, re.M)),
        "VERSION": version.group(1) if version else "",
    }


def cq_rows():
    rows = []
    for q in sorted((ROOT / "queries").glob("CQ*.rq")):
        text = q.read_text(encoding="utf-8")
        m = re.match(r"#\s*(CQ\w+):\s*(.+)", text)
        if not m:
            continue
        examples = re.findall(r"^#\s+-\s+(.+)$", text, re.M)[:2]
        rows.append((m.group(1), m.group(2).strip(), examples))
    rows.sort(key=lambda r: (int(re.search(r"\d+", r[0]).group()), r[0]))
    return rows


def cq_table():
    rows = cq_rows()
    out = ['<table class="ffo-table"><tr><th>CQ</th><th>Competency question</th>'
           '<th>Example ForestFire-VQA questions</th></tr>']
    for cid, title, ex in rows:
        out.append(f"<tr><td><b>{cid}</b></td><td>{html.escape(title)}</td>"
                   f"<td>{'<br>'.join(html.escape(e) for e in ex)}</td></tr>")
    out.append("</table>")
    return "\n".join(out), len(rows)


def _term(cell):
    """ROBOT TSV cell -> plain value: <iri> -> iri, "text"@en -> text."""
    c = (cell or "").strip()
    if c.startswith("<") and c.endswith(">"):
        return c[1:-1]
    m = re.match(r'^"(.*)"(?:@[\w-]+|\^\^\S+)?$', c, re.S)
    if m:
        return m.group(1).replace('\\"', '"')
    return re.sub(r"(?:@[A-Za-z]{2,3}(?:-[\w]+)?|\^\^\S+)$", "", c)    # ROBOT writes Label@en without quotes


def local(iri):
    return re.split(r"[#/]", iri)[-1]


def vocab_tree(tsv_path):
    if not tsv_path or not Path(tsv_path).exists():
        return "<p>(concept hierarchy not generated – ROBOT query output missing)</p>"
    with open(tsv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    head = [h.lstrip("?") for h in rows[0]]
    recs = [dict(zip(head, (_term(c) for c in r))) for r in rows[1:]]
    schemes = {}
    info, parents = {}, defaultdict(set)
    members = defaultdict(set)
    for r in recs:
        s, c = r["scheme"], r["concept"]
        schemes[s] = r.get("schemeLabel") or local(s)
        members[s].add(c)
        info[c] = {"label": r["label"], "kw": r.get("keywords", ""), "def": r.get("definition", "")}
        if r.get("broader"):
            parents[c].add(r["broader"])
    out = ['<div class="ffo-tree">']
    for s in sorted(schemes, key=lambda x: schemes[x].lower()):
        mem = members[s]
        children = defaultdict(list)
        roots = []
        for c in mem:
            ps = [p for p in parents.get(c, ()) if p in mem]
            if ps:
                for p in ps:
                    children[p].append(c)
            else:
                roots.append(c)
        key = lambda c: info[c]["label"].lower()

        def render(c, depth=0, seen=()):
            i = info[c]
            kw = f' <span class="ffo-kw">({html.escape(i["kw"])})</span>' if i["kw"] else ""
            df = f'<br><span class="ffo-def">{html.escape(i["def"])}</span>' if i["def"] else ""
            line = f'<li><a href="#{html.escape(local(c))}" title="{html.escape(c)}">{html.escape(i["label"])}</a>{kw}{df}'
            kids = [k for k in sorted(children.get(c, []), key=key) if k not in seen]
            if kids and depth < 8:
                line += "<ul>" + "".join(render(k, depth + 1, seen + (c,)) for k in kids) + "</ul>"
            return line + "</li>"

        out.append(f'<details class="ffo-scheme" open><summary>{html.escape(schemes[s])} '
                   f'<span class="ffo-kw">({len(mem)} concepts)</span></summary><ul>')
        out += [render(r) for r in sorted(roots, key=key)]
        out.append("</ul></details>")
    out.append("</div>")
    return "\n".join(out)


# ------------------------------------------------------------------ output
def fill(name, values):
    text = (HERE / "sections" / name).read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    left = re.findall(r"\{\{\w+\}\}", text)
    if left:
        raise SystemExit(f"{name}: unfilled placeholders {left}")
    return text


def install(doc_dir, sections):
    sec = doc_dir / "sections"
    if not sec.is_dir():
        raise SystemExit(f"{sec} not found – did WIDOCO run (without -uniteSections)?")
    for fname, text in sections.items():
        (sec / fname).write_text(text, encoding="utf-8")
    idx = doc_dir / "index-en.html"
    page = idx.read_text(encoding="utf-8")
    if "ffo-table{" not in page:                          # our CSS, once
        page = page.replace("</head>", EXTRA_CSS + "</head>", 1)
        idx.write_text(page, encoding="utf-8")
    shutil.copyfile(idx, doc_dir / "index.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site")
    ap.add_argument("--vocab-tsv", default=None)
    a = ap.parse_args()
    site = Path(a.site)
    stats = ttl_stats()
    table, n_cq = cq_table()
    values = {**stats, "N_CQ": n_cq, "CQ_TABLE": table}

    install(site, {"introduction-en.html": fill("core-introduction.html", values),
                   "description-en.html": fill("core-description.html", values),
                   "references-en.html": fill("core-references.html", values)})
    if (site / "vocab").is_dir():
        install(site / "vocab", {"introduction-en.html": fill("vocab-introduction.html", values),
                                 "description-en.html": fill("vocab-description.html",
                                                             {**values, "VOCAB_TREE": vocab_tree(a.vocab_tsv)})})
    if (site / "rules").is_dir():
        install(site / "rules", {"introduction-en.html": fill("rules-introduction.html", values)})

    res = site / "resources"
    res.mkdir(exist_ok=True)
    for f in ("ontology/ffonto-core.ttl", "ontology/ffonto-vocab.ttl", "ontology/danger-rules.ttl",
              "ontology/ffonto.ttl", "ontology/ffonto-shapes.ttl", "context/ffonto.context.jsonld"):
        shutil.copyfile(ROOT / f, res / Path(f).name)
    (site / ".nojekyll").write_text("", encoding="utf-8")
    print(f"sections installed: {n_cq} CQs, {stats['N_CLASSES']} classes, {stats['N_OBJ']} object + "
          f"{stats['N_DATA']} datatype properties, {stats['N_CONCEPTS']} concepts in {stats['N_SCHEMES']} schemes")


if __name__ == "__main__":
    main()
