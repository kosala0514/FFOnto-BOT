#!/usr/bin/env python3
"""
oops_check.py — send FFOnto to the OOPS! (OntOlogy Pitfall Scanner!) web service
and summarise the pitfalls it finds.

    python validation/oops_check.py            # core schema only (recommended)
    python validation/oops_check.py --all      # core + vocabulary + danger rules

Outputs build/oops_response.xml (raw) and build/oops_report.md (summary).
OOPS! only accepts RDF/XML, so the Turtle files are converted with rdflib first.
Web service docs: https://oops.linkeddata.es/webservice.html
If the service is down, paste build/ffonto-core.owl into https://oops.linkeddata.es instead.
"""
import argparse, sys, urllib.request, xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from rdflib import Graph

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://oops.linkeddata.es/rest"

# Pitfalls we expect and accept by design – documented so the report is easy to discuss.
EXPECTED = {
    "P13": "Inverse properties are deliberately not declared (one direction per relation keeps the KG small; SPARQL uses ^path).",
    "P11": "Remaining cases are deliberate: reused SOSA/PROV properties (redefining their domain/range would hijack those vocabularies), ffonto:threatens (points to entities or concepts) and ffonto:onProperty (its values are properties; a range of owl:DatatypeProperty is not allowed in OWL 2 DL).",
    "P04": "Reused external classes (SOSA, PROV-O, SKOS) are declared only so FFOnto works offline; they are connected through FFOnto subclasses.",
    "P08": "Any remaining items are reused external terms, documented in their own specifications.",
}


def local(tag):
    return tag.rsplit("}", 1)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include vocabulary and danger rules")
    ap.add_argument("--pitfalls", default="", help="comma-separated pitfall numbers, empty = all")
    a = ap.parse_args()

    files = ["ffonto-core.ttl"] + (["ffonto-vocab.ttl", "danger-rules.ttl"] if a.all else [])
    g = Graph()
    for f in files:
        g.parse(ROOT / "ontology" / f, format="turtle")
    rdfxml = g.serialize(format="xml")
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "oops_input.owl").write_text(rdfxml, encoding="utf-8")

    body = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<OOPSRequest>\n"
            "  <OntologyUrl></OntologyUrl>\n"
            f"  <OntologyContent><![CDATA[{rdfxml}]]></OntologyContent>\n"
            f"  <Pitfalls>{a.pitfalls}</Pitfalls>\n"
            "  <OutputFormat>XML</OutputFormat>\n</OOPSRequest>")
    req = urllib.request.Request(ENDPOINT, data=body.encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/xml"})
    print(f"sending {len(g)} triples ({', '.join(files)}) to {ENDPOINT} …")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            xml = r.read().decode("utf-8")
    except Exception as e:
        sys.exit(f"OOPS! request failed: {e}\nUse the web form instead: https://oops.linkeddata.es "
                 f"(paste build/oops_input.owl)")
    (ROOT / "build" / "oops_response.xml").write_text(xml, encoding="utf-8")

    root = ET.fromstring(xml)
    pitfalls = []
    for el in root.iter():
        if local(el.tag) == "Pitfall":
            info = defaultdict(list)
            for child in el.iter():
                name = local(child.tag)
                if child is not el and child.text and child.text.strip():
                    info[name].append(child.text.strip())
            pitfalls.append(info)

    lines = ["# OOPS! pitfall report\n", f"Input: {', '.join(files)} ({len(g)} triples)\n",
             "| Code | Importance | Name | # affected | Expected by design? |", "|---|---|---|---|---|"]
    order = {"Critical": 0, "Important": 1, "Minor": 2}
    pitfalls.sort(key=lambda p: (order.get((p.get("Importance") or ["?"])[0], 3), (p.get("Code") or [""])[0]))
    for p in pitfalls:
        code = (p.get("Code") or ["?"])[0]
        imp = (p.get("Importance") or ["?"])[0]
        name = (p.get("Name") or ["?"])[0]
        n = (p.get("NumberAffectedElements") or [str(len(p.get("AffectedElement", [])))])[0]
        lines.append(f"| {code} | {imp} | {name} | {n} | {EXPECTED.get(code, '')} |")
    lines.append("\n## Affected elements\n")
    for p in pitfalls:
        code = (p.get("Code") or ["?"])[0]
        affected = p.get("AffectedElement", [])
        if affected:
            lines.append(f"**{code}**: " + ", ".join(x.rsplit('#', 1)[-1] for x in affected[:40])
                         + (" …" if len(affected) > 40 else ""))
    (ROOT / "build" / "oops_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:2 + 2 + len(pitfalls)]))
    if not pitfalls:
        print("No pitfalls parsed – open build/oops_response.xml to check the raw answer.")
    print("\nwrote build/oops_report.md and build/oops_response.xml")


if __name__ == "__main__":
    main()
