#!/usr/bin/env python3
"""
make_prompt.py — build the scene-extraction prompt for a vision-language model (VLM)
from the SKOS vocabulary, so the VLM can only answer with terms the ontology knows.

    python scripts/make_prompt.py  >  prompts/scene_extraction_prompt.md

Re-run it whenever you add concepts to ontology/ffonto-vocab.ttl.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
txt = (ROOT / "ontology" / "ffonto-vocab.ttl").read_text(encoding="utf-8")

# parse concept blocks: name, scheme, broader
concepts = {}
for block in re.split(r"\n(?=ffv:\w+ a skos:Concept\s*;)", txt):
    m = re.match(r"ffv:(\w+) a skos:Concept\s*;", block)
    if not m:
        continue
    name = m.group(1)
    scheme = re.search(r"skos:inScheme ffv:(\w+)", block).group(1)
    broader = re.search(r"skos:broader ffv:(\w+)", block)
    concepts[name] = (scheme, broader.group(1) if broader else None)

def in_scheme(s):
    return [n for n, (sc, _) in concepts.items() if sc == s]

def leaves_under(top):
    out = []
    for n, (sc, b) in concepts.items():
        x = n
        while x:
            if x == top:
                out.append(n); break
            x = concepts.get(x, (None, None))[1]
    return [n for n in out if n != top]

def children(parent):
    return [n for n, (_, b) in concepts.items() if b == parent]

kinds = {top: leaves_under(top) for top in in_scheme("EntityKindScheme") if concepts[top][1] is None}
values = {g: children(g) for g in ["Colour", "Level", "Direction", "Sky", "Light", "FireTypeValue", "SpreadStatus", "Purpose", "Distribution"]}

print("""# Scene-graph extraction prompt (generated from ffonto-vocab.ttl)

Use this as the system / user prompt for a vision-language model (e.g. Qwen2-VL, LLaVA, GPT-4o, Gemini).
Ask for JSON only. Save each answer as `data/scenes/<image_id>.json`, then run `python scripts/lift.py`.

---

You are annotating ONE wildfire image for a knowledge graph. Describe only what is VISIBLE.
Return ONLY a JSON object with this structure:

```json
{
  "scene_id": "scene_<image_id>",
  "image_record_uuid": "<record_uuid of the image record>",
  "extractor": {"model_name": "<your model name>", "model_version": "<version>"},
  "human_validated": false,
  "entities": [
    {"id": "e1", "kind": "<KIND>", "count": 1,
     "state": ["<STATE>", ...], "activity": ["<ACTIVITY>", ...], "wears": ["<PPE>", ...],
     "position": "<POSITION>",
     "near": ["e2"], "advancing_toward": [], "reached": [], "protects": [], "blocks": []}
  ],
  "attributes": {"<PROPERTY>": "<VALUE>"}
}
```

Rules:
- Use ONLY the exact names listed below. If something does not fit, leave it out.
- One entity per distinct object or group; use "count" for groups (e.g. 3 firefighters).
- Relations ("near", "advancing_toward", "reached", "protects", "blocks") refer to entity ids.
- Positions are frame-relative to the fire.
""")
print("## KIND (entity kinds)\n")
for top, ls in kinds.items():
    print(f"- **{top.replace('Kind', '')}**: " + ", ".join(ls))
print("\n## STATE\n")
for g in ["DamageState", "FireState", "AccessState", "OperationalState", "VegetationState"]:
    print(f"- **{g}**: " + ", ".join(children(g)))
print("\n## ACTIVITY\n\n" + ", ".join(n for n in in_scheme("ActivityScheme") if children(n) == []))
print("\n## PPE\n\n" + ", ".join(in_scheme("PPEScheme")))
print("\n## POSITION\n\n" + ", ".join(in_scheme("PositionScheme")))
print("\n## PROPERTY → allowed VALUE\n")
pv = {
  "SmokeColour": "Colour", "AshColour": "Colour", "SmokeDensity": "Level", "PlumeHeight": "Level",
  "VisibilityLevel": "Level", "AirQualityLevel": "Level", "FireIntensity": "Level", "BurnSeverity": "Level",
  "VegetationDensity": "Level", "PlumeDirection": "Direction", "DominantAspect": "Direction",
  "SkyCondition": "Sky", "LightCondition": "Light", "FireType": "FireTypeValue",
  "FireSpreadStatus": "SpreadStatus", "FireActivityTrend": "SpreadStatus", "ScenePurpose": "Purpose",
  "VegetationDistribution": "Distribution",
}
values["Compass"] = ["North", "East", "South", "West"]
values["Direction"] = ["Vertical", "Drifting"]
pv["DominantAspect"] = "Compass"
for p, grp in pv.items():
    print(f"- **{p}**: " + ", ".join(values[grp]))
