#!/usr/bin/env python3
"""
lift.py — turn the raw multimodal records into RDF using the FFOnto JSON-LD context.

    raw image / audio JSON  ─┐
    weather CSV             ─┼─► normalise() ─► JSON-LD (+ context) ─► RDF graph (data/kg.ttl)
    scene-graph JSON (VLM)  ─┘

Why a normaliser before JSON-LD?
  The raw JSON is *almost* JSON-LD-ready. Three things cannot be expressed by a
  @context alone, so this script fixes them first (nothing else is changed):
    1. nested wrappers (provenance, quality, image_size_px) are flattened;
    2. score maps such as {"smoke": 0.73, "fire": 0.16} and threshold maps such as
       {"fire_threshold": 0.55} become explicit observation / threshold nodes;
    3. identifiers are minted (record, event, model, site) and class names such as
       "fire" or "Firefighter" are turned into vocabulary IRIs (ffontov:Fire, ffontov:Firefighter).
  Every key left in the normalised JSON is mapped by context/ffonto.context.jsonld;
  the script warns about any key the context does not cover (JSON-LD would drop it silently).

Usage (from the repository root):
    python scripts/lift.py                 # writes data/jsonld/*.jsonld, data/graphs/<modality>.ttl, data/kg.ttl
    python scripts/lift.py --no-rdf        # only writes JSON-LD (no rdflib needed)
"""
import argparse, csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CTX_FILE = ROOT / "context" / "ffonto.context.jsonld"
VOCAB_FILE = ROOT / "ontology" / "ffonto-vocab.ttl"

WARNINGS = []
def warn(msg):
    WARNINGS.append(msg)

# ---------------------------------------------------------------- vocabulary
def load_concepts():
    """Concept local names in ffonto-vocab.ttl (regex – no rdflib needed)."""
    txt = VOCAB_FILE.read_text(encoding="utf-8")
    return set(re.findall(r"^ffontov:(\w+) a skos:Concept\s*;", txt, flags=re.M))

CONCEPTS = load_concepts()

# raw class / label strings  ->  concept local name
LABEL_MAP = {
    # image classifier classes
    "fire": "Fire", "smoke": "Smoke", "no fire": "NoFire", "no_fire": "NoFire",
    # label sources
    "caption": "CaptionSource", "zshot": "ZeroShotSource", "zero-shot": "ZeroShotSource",
    "detector": "DetectorSource", "human": "HumanSource", "vlm": "VLMSource", "audio": "AudioTaggerSource",
    # severities
    "low": "Low", "medium": "Medium", "high": "High", "extreme": "Extreme",
    # audio (YAMNet / AudioSet display names -> vocab)
    "explosion": "Explosion", "crackle": "Crackling", "crackling": "Crackling", "fire crackle": "Crackling",
    "siren": "Siren", "civil defense siren": "Siren", "emergency vehicle": "Siren",
    "speech": "Speech", "vehicle": "VehicleSound", "engine": "VehicleSound",
}

def concept(raw, what="value"):
    """Map a raw label / short name to a compact vocab IRI 'ffontov:X'. Returns None if unknown."""
    if raw is None or raw == "":
        return None
    key = str(raw).strip()
    name = LABEL_MAP.get(key.lower(), key)
    if name.startswith("ffontov:"):
        name = name.split(":", 1)[1]
    if name not in CONCEPTS:
        warn(f"unknown {what} '{raw}' – add it to ontology/ffonto-vocab.ttl or LABEL_MAP; skipped")
        return None
    return "ffontov:" + name

def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-").lower()

def model_node(name, version=None, **extra):
    node = {"@id": f"ffd:model/{slug(name)}" + (f"/{slug(version)}" if version else ""),
            "@type": "ffonto:Model", "model_name": name}
    if version is not None:
        node["model_version"] = str(version)
    node.update({k: v for k, v in extra.items() if v is not None})
    return node

def event_node(ev):
    if not ev:
        return None
    return {"@id": f"ffd:event/{slug(ev)}", "@type": "ffonto:FireEvent", "event_code": ev}

def site_node(name, **extra):
    if not name:
        return None
    n = {"@id": f"ffd:site/{slug(name)}", "@type": "ffonto:Site", "site_name": name}
    n.update({k: v for k, v in extra.items() if v not in (None, "")})
    return n

def geometry_wkt(g):
    """Accept [lon, lat], {"lon":..,"lat":..}, a WKT string or GeoJSON; return WKT or JSON text."""
    if g is None:
        return None
    if isinstance(g, str):
        return g
    if isinstance(g, (list, tuple)) and len(g) == 2 and all(isinstance(x, (int, float)) for x in g):
        return f"POINT({g[0]} {g[1]})"
    if isinstance(g, dict) and {"lon", "lat"} <= set(g):
        return f"POINT({g['lon']} {g['lat']})"
    return json.dumps(g)

def iso_time(v):
    """Return a valid ISO-8601 timestamp string, or None (so 'None', '' or junk never become a fake time)."""
    if v is None or str(v).strip().lower() in {"", "none", "null", "nan"}:
        return None
    import datetime as _dt
    t = str(v).strip().replace("Z", "+00:00")
    try:
        _dt.datetime.fromisoformat(t)
    except ValueError:
        warn(f"invalid capture_time_utc {v!r} – skipped")
        return None
    return str(v).strip()

def num(v):
    if v is None or (isinstance(v, str) and v.strip().lower() in {"", "none", "null", "nan", "na", "n/a", "-"}):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        warn(f"not a number: {v!r} – value skipped")
        return None

def drop_nulls(d):
    return {k: v for k, v in d.items() if v is not None and v != "" and v != []}

# ---------------------------------------------------------------- image
def normalise_image(raw):
    rid = f"ffd:record/{raw['record_uuid']}"
    prov = raw.get("provenance") or {}
    quality = raw.get("quality") or {}
    size = raw.get("image_size_px") or {}

    # collect thresholds from any label: {"fire_threshold": .55} -> {"Fire": .55}
    thresholds = {}
    for lab in raw.get("labels", []):
        for k, v in (lab.get("thresholds") or {}).items():
            c = concept(k.replace("_threshold", ""), "threshold class")
            if c:
                thresholds[c] = v

    def threshold_node(c):
        if c not in thresholds:
            return None
        return {"@id": f"{rid}/threshold/{c.split(':', 1)[1]}", "@type": "ffonto:Threshold",
                "for_class": c, "threshold_value": num(thresholds[c])}

    observations, seen_classes = [], set()
    for i, lab in enumerate(raw.get("labels", [])):
        c = concept(lab.get("type"), "image label class")
        if not c:
            continue
        seen_classes.add(c)
        if lab.get("opacity") is not None:
            warn(f"{rid}: label opacity={lab['opacity']} is not modelled yet")
        observations.append(drop_nulls({
            "@id": f"{rid}/obs/{i}", "@type": "ffonto:ModelObservation",
            "feature_of_interest": rid,
            "predicted_class": c,
            "confidence": num(lab.get("confidence")),
            "severity": concept(lab.get("severity"), "severity"),
            "label_source": concept(lab.get("label_source"), "label source"),
            "attributed_to": model_node(lab["model_name"], lab.get("model_version")) if lab.get("model_name") else None,
            "used_threshold": threshold_node(c),
        }))
    # zero-shot scores not already present as labels (e.g. "no fire")
    zmodel = prov.get("zshot_model")
    versions = {l.get("model_name"): l.get("model_version") for l in raw.get("labels", []) if l.get("model_name")}
    for j, (cls, score) in enumerate((prov.get("zshot_scores") or {}).items()):
        c = concept(cls, "zero-shot class")
        if not c or c in seen_classes:
            continue
        observations.append(drop_nulls({
            "@id": f"{rid}/obs/z{j}", "@type": "ffonto:ModelObservation",
            "feature_of_interest": rid, "predicted_class": c, "confidence": num(score),
            "label_source": "ffontov:ZeroShotSource",
            "attributed_to": model_node(zmodel, versions.get(zmodel)) if zmodel else None,
            "used_threshold": threshold_node(c),
        }))

    caption = None
    if prov.get("caption_text"):
        caption = drop_nulls({
            "@id": f"{rid}/caption", "@type": "ffonto:Caption",
            "caption_text": prov["caption_text"],
            "attributed_to": model_node(prov["caption_model"]) if prov.get("caption_model") else None,
        })

    return drop_nulls({
        "@id": rid, "@type": "ffonto:ImageRecord",
        "schema_version": raw.get("schema_version"),
        "record_uuid": raw["record_uuid"],
        "image_id": raw.get("image_id"),
        "event_id": event_node(raw.get("event_id")),
        "source": raw.get("source"),
        "capture_time_utc": iso_time(raw.get("capture_time_utc")),
        "crs": raw.get("crs"),
        "center": geometry_wkt(raw.get("center")),
        "footprint": geometry_wkt(raw.get("footprint")),
        "resolution_m": num(raw.get("resolution_m")),
        "bands": ",".join(raw["bands"]) if isinstance(raw.get("bands"), list) else raw.get("bands"),
        "georeferenced": raw.get("georef") is not None,
        "cloud_cover": num(quality.get("cloud_cover")),
        "blur": num(quality.get("blur")),
        "uri": prov.get("uri"),
        "checksum_sha256": prov.get("checksum_sha256"),
        "license": raw.get("license"),
        "width": size.get("width"),
        "height": size.get("height"),
        "caption": caption,
        "observations": observations,
    })

# ---------------------------------------------------------------- audio
def normalise_audio(raw):
    rid = f"ffd:record/{raw['record_uuid']}"
    prov = raw.get("provenance") or {}
    url = prov.get("model")
    m = re.search(r"tfhub\.dev/([^/]+/[^/]+)/(\w+)", url or "")
    model = model_node(m.group(1) if m else (url or "unknown-audio-model"), m.group(2) if m else None,
                       model_url=url, class_map_sha1=prov.get("class_map_sha1"),
                       frame_stride_s=num(prov.get("frame_stride_s")), top_k=prov.get("topk"))
    observations = []
    for i, lab in enumerate(raw.get("labels", [])):
        c = concept(lab.get("raw_label") or lab.get("type"), "audio label")
        if not c:
            continue
        thr = (lab.get("thresholds") or {}).get("frame_thresh")
        observations.append(drop_nulls({
            "@id": f"{rid}/obs/{i}", "@type": "ffonto:ModelObservation",
            "feature_of_interest": rid,
            "predicted_class": c,
            "label_category": lab.get("type"),
            "confidence": num(lab.get("mean_confidence")),      # convention: confidence = mean confidence
            "mean_confidence": num(lab.get("mean_confidence")),
            "max_confidence": num(lab.get("max_confidence")),
            "top_k_labels": lab.get("topk_frame_labels"),
            "label_source": "ffontov:AudioTaggerSource",
            "attributed_to": model,
            "used_threshold": {"@id": f"{rid}/threshold/{c.split(':', 1)[1]}", "@type": "ffonto:Threshold",
                               "for_class": c, "threshold_value": num(thr)} if thr is not None else None,
        }))
    return drop_nulls({
        "@id": rid, "@type": "ffonto:AudioRecord",
        "schema_version": raw.get("schema_version"),
        "record_uuid": raw["record_uuid"],
        "event_id": event_node(raw.get("event_id")),
        "audio_file": raw.get("audio_file"),
        "source": raw.get("source"),
        "device_id": raw.get("device_id"),
        "at_site": site_node(raw.get("site")),
        "region": raw.get("region"),
        "capture_time_utc": iso_time(raw.get("capture_time_utc")),
        "crs": raw.get("crs"),
        "center": geometry_wkt(raw.get("center")),
        "duration_ms": num(raw.get("duration_ms")),
        "global_threshold": num(prov.get("thresh")),
        "observations": observations,
    })

# ---------------------------------------------------------------- weather
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                                       "sep", "oct", "nov", "dec"], start=1)}
MISSING = {"", "none", "null", "nan", "na", "n/a", "-"}


def clean(v):
    """Return None for empty / 'None' / 'nan' cells, else the stripped string."""
    if v is None:
        return None
    v = str(v).strip()
    return None if v.lower() in MISSING else v


def as_int(v):
    v = clean(v)
    if v is None:
        return None
    if v.lower()[:3] in MONTHS:                       # month given as a name ("aug")
        return MONTHS[v.lower()[:3]]
    try:
        return int(float(v))
    except ValueError:
        return None


def weather_date(row):
    """Build 'YYYY-MM-DDT00:00:00Z' from a 'date' column or year/month/day columns.
    Returns None (never a 'None…' string) when the row has no complete date."""
    d = clean(row.get("date"))
    if d and re.match(r"^\d{4}-\d{2}-\d{2}", d):
        return d[:10] + "T00:00:00Z"
    y, m, day = as_int(row.get("year")), as_int(row.get("month_num") or row.get("month")), as_int(row.get("day_of_month"))
    if y and m and day:
        try:
            import datetime as _dt
            _dt.date(y, m, day)                          # rejects impossible dates such as 2012-02-31
            return f"{y:04d}-{m:02d}-{day:02d}T00:00:00Z"
        except ValueError:
            return None
    return None


def normalise_weather(row, index=0):
    row = {k.strip(): v for k, v in row.items() if k is not None}
    date = weather_date(row)                             # xsd:date is not in OWL 2 -> stored as xsd:dateTime
    gx, gy = as_int(row.get("grid_x")), as_int(row.get("grid_y"))
    site = clean(row.get("site")) or "unknown-site"
    # A site node must have ONE grid cell, otherwise every query that touches gridX/gridY
    # multiplies rows. If the table has grid coordinates, each grid cell becomes its own site.
    site_key = f"{slug(site)}/grid-{gx}-{gy}" if gx is not None and gy is not None else slug(site)
    site_label = f"{site} (grid {gx},{gy})" if gx is not None and gy is not None else site
    # the record IRI must be unique per row: use the date if there is one, otherwise the row number
    rid = f"ffd:record/weather/{site_key}/{date[:10] if date else f'row-{index:06d}'}"
    if date is None:
        warn(f"weather row {index}: no complete date (year/month_num/day_of_month) – stored without ffonto:observationDate")
    fo = clean(row.get("fire_occured") or row.get("fire_occurred"))
    return drop_nulls({
        "@id": rid, "@type": "ffonto:WeatherRecord",
        "source": "weather_csv",
        "event_id": event_node(clean(row.get("event_id"))),
        "at_site": {**site_node(site_label, country=clean(row.get("country")), grid_x=gx, grid_y=gy),
                    "@id": f"ffd:site/{site_key}"},
        "date": date, "day_name": clean(row.get("day_name") or row.get("day")),
        "temp_c": num(row.get("temp_c")), "rh_pct": num(row.get("rh_pct")),
        "wind_kmh": num(row.get("wind_kmh")), "rain_mm": num(row.get("rain_mm")),
        "FFMC": num(row.get("FFMC")), "DMC": num(row.get("DMC")), "DC": num(row.get("DC")),
        "ISI": num(row.get("ISI")), "BUI": num(row.get("BUI")), "FWI": num(row.get("FWI")),
        "fire_occured": (fo.lower() in ("1", "1.0", "true", "yes")) if fo is not None else None,
        "area_ha": num(row.get("area_ha")),
    })

# ---------------------------------------------------------------- scene graph (VLM output)
def normalise_scene(raw):
    sid = f"ffd:scene/{slug(raw['scene_id'])}"
    rec = f"ffd:record/{raw['image_record_uuid']}"
    ex = raw.get("extractor") or {}
    model = model_node(ex.get("model_name", "unknown-extractor"), ex.get("model_version"))
    eid = lambda local: f"{sid}/{slug(local)}"

    entities = []
    for e in raw.get("entities", []):
        kind = concept(e.get("kind"), "entity kind")
        if not kind:
            continue
        cl = lambda key, what: [c for c in (concept(x, what) for x in (e.get(key) or [])) if c]
        entities.append(drop_nulls({
            "@id": eid(e["id"]),
            "kind": kind,
            "count": int(e["count"]) if e.get("count") is not None else None,
            "state": cl("state", "state"),
            "activity": cl("activity", "activity"),
            "wears": cl("wears", "PPE item"),
            "position": concept(e.get("position"), "position"),
            "near": [eid(x) for x in e.get("near", [])],
            "advancing_toward": [eid(x) for x in e.get("advancing_toward", [])],
            "reached": [eid(x) for x in e.get("reached", [])],
            "protects": [eid(x) for x in e.get("protects", [])],
            "blocks": [eid(x) for x in e.get("blocks", [])],
        }))

    attributes = []
    for i, (prop, val) in enumerate((raw.get("attributes") or {}).items()):
        p, v = concept(prop, "observable property"), concept(val, f"value for {prop}")
        if p and v:
            attributes.append({"@id": f"{sid}/attr/{i}", "@type": "ffonto:ModelObservation",
                               "feature_of_interest": rec, "observed_property": p, "value": v,
                               "label_source": "ffontov:VLMSource", "attributed_to": model["@id"]})
    return drop_nulls({
        "@id": sid, "@type": "ffonto:SceneDescription",
        "describes_record": rec,
        "attributed_to": model,
        "human_validated": raw.get("human_validated"),
        "entities": entities,
        "attributes": attributes,
    })

MODALITY_OF_TYPE = {"ffonto:ImageRecord": "image", "ffonto:AudioRecord": "audio",
                    "ffonto:WeatherRecord": "weather", "ffonto:SceneDescription": "scene"}

# ---------------------------------------------------------------- context coverage check
def check_keys(obj, ctx_terms, where):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not k.startswith("@") and k not in ctx_terms:
                warn(f"{where}: key '{k}' is not in the JSON-LD context and would be dropped")
            check_keys(v, ctx_terms, where)
    elif isinstance(obj, list):
        for v in obj:
            check_keys(v, ctx_terms, where)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--scenes", default=str(ROOT / "data" / "scenes"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--no-rdf", action="store_true", help="only write JSON-LD files")
    a = ap.parse_args()

    ctx = json.loads(CTX_FILE.read_text(encoding="utf-8"))
    ctx_terms = set(ctx["@context"].keys())
    out_dir = Path(a.out) / "jsonld"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    for p in sorted(Path(a.raw).glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        mod = str(raw.get("modality", "")).lower()
        if mod == "image":
            docs.append((p.stem, normalise_image(raw)))
        elif mod == "audio":
            docs.append((p.stem, normalise_audio(raw)))
        else:
            warn(f"{p.name}: unknown modality '{raw.get('modality')}' – skipped")
    for p in sorted(Path(a.raw).glob("*.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                docs.append((f"{p.stem}_{i:05d}", normalise_weather(row, i)))
    if Path(a.scenes).exists():
        for p in sorted(Path(a.scenes).glob("*.json")):
            docs.append((p.stem, normalise_scene(json.loads(p.read_text(encoding="utf-8")))))

    for name, doc in docs:
        check_keys(doc, ctx_terms, name)
        (out_dir / f"{name}.jsonld").write_text(
            json.dumps({"@context": ctx["@context"], **doc}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(docs)} JSON-LD documents to {out_dir}")

    if not a.no_rdf:
        try:
            from rdflib import Graph
        except ImportError:
            sys.exit("rdflib not installed: pip install rdflib   (or use --no-rdf)")
        g = Graph()
        g.bind("ffonto", "http://purl.org/ffonto#"); g.bind("ffontov", "http://purl.org/ffonto/vocab#")
        g.bind("ffd", "http://purl.org/ffonto/data/"); g.bind("sosa", "http://www.w3.org/ns/sosa/")
        g.bind("prov", "http://www.w3.org/ns/prov#")
        # one graph per modality (for GraphDB named graphs) + one merged graph
        parts = {}
        for name, doc in docs:
            mod = MODALITY_OF_TYPE.get(doc.get("@type"), "other")
            gp = parts.setdefault(mod, Graph())
            gp.parse(out_dir / f"{name}.jsonld", format="json-ld")
        gdir = Path(a.out) / "graphs"
        gdir.mkdir(exist_ok=True)
        for mod, gp in parts.items():
            for pfx, ns in g.namespaces():
                gp.bind(pfx, ns)
            gp.serialize(gdir / f"{mod}.ttl", format="turtle")
            g += gp
            print(f"  {mod:<8} {len(gp):>6} triples -> {gdir / (mod + '.ttl')}")
        kg = Path(a.out) / "kg.ttl"
        g.serialize(kg, format="turtle")
        print(f"wrote {len(g)} triples to {kg}")

    for w in WARNINGS[:20]:
        print("WARNING:", w)
    if len(WARNINGS) > 20:
        print(f"WARNING: … and {len(WARNINGS) - 20} more (see data/lift_warnings.txt)")
    (Path(a.out) / "lift_warnings.txt").write_text("\n".join(WARNINGS), encoding="utf-8")

if __name__ == "__main__":
    main()
