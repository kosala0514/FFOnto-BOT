"""
Visual outputs for knowledge-graph answers: Vega-Lite chart specs built from the SPARQL result rows.

The chart is drawn from the SAME rows the answer was generated from, so the picture is as traceable as the text.
Specs are plain JSON (Vega-Lite v5): Streamlit renders them with st.vega_lite_chart, any web client with vega-embed.

    chart_for(cq_id, columns, rows) -> list of {"kind": "vega", "title", "spec", "source"}
"""
import re

from . import config

VL = "https://vega.github.io/schema/vega-lite/v5.json"
MAX_POINTS = 400

FWI_COLS = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
WEATHER_COLS = ["tempC", "rhPct", "windKmh", "rainMm"]
ID_LIKE = re.compile(r"^(record|w|obs|output|event|eventCode|iri|s|x|gridX|gridY)$", re.I)


def num(v):
    try:
        f = float(str(v).strip())
        return f if f == f else None             # drop NaN
    except (TypeError, ValueError):
        return None


def label(v):
    """'ffv:CrownFire' -> 'CrownFire'; long IRIs -> local name."""
    v = str(v or "")
    v = re.sub(r"^[A-Za-z]+:", "", v) if not v.startswith("http") else re.split(r"[#/]", v)[-1]
    return v


def numeric_cols(cols, rows):
    out = []
    for c in cols:
        vals = [r.get(c, "") for r in rows if str(r.get(c, "")).strip()]
        if vals and not ID_LIKE.match(c) and sum(num(v) is not None for v in vals) >= 0.8 * len(vals):
            out.append(c)
    return out


def _date_col(cols, rows):
    for c in cols:
        if c.lower() in ("date", "time", "datetime", "capturetime"):
            if len({str(r.get(c, ""))[:10] for r in rows if r.get(c)}) >= 2:
                return c
    return None


def _spec(title, data, **body):
    return {"$schema": VL, "title": title, "data": {"values": data[:MAX_POINTS]}, "width": "container", **body}


# ------------------------------------------------------------------ chart builders
def _scores_vs_threshold(rows, title):
    data = [{"record": label(r.get("record")), "class": label(r.get("class")), "confidence": num(r.get("confidence")),
             "threshold": num(r.get("threshold")),
             "above": "above threshold" if str(r.get("aboveThreshold", "")).lower() in ("true", "1") else "below threshold"}
            for r in rows if num(r.get("confidence")) is not None]
    if not data:
        return None
    layer = [
        {"mark": {"type": "bar", "tooltip": True},
         "encoding": {"x": {"field": "class", "type": "nominal", "title": None, "sort": "-y"},
                      "y": {"field": "confidence", "type": "quantitative", "scale": {"domain": [0, 1]}},
                      "color": {"field": "above", "type": "nominal", "title": None,
                                "scale": {"domain": ["above threshold", "below threshold"],
                                          "range": ["#d9480f", "#adb5bd"]}}}},
        {"mark": {"type": "tick", "thickness": 3, "color": "#1c1c1c", "tooltip": True},
         "encoding": {"x": {"field": "class", "type": "nominal"},
                      "y": {"field": "threshold", "type": "quantitative"}}},
    ]
    records = sorted({d["record"] for d in data})
    if len(records) > 1:
        data = [d for d in data if d["record"] in records[:4]]
        return {"$schema": VL, "title": title, "data": {"values": data},
                "facet": {"column": {"field": "record", "type": "nominal", "title": None}},
                "spec": {"layer": layer, "width": 220}}
    return _spec(title, data, layer=layer)


def _time_series(rows, cols, date_col, title):
    sites = sorted({label(r.get("site", "")) for r in rows})
    data = []
    for r in rows:
        for c in cols:
            v = num(r.get(c))
            if v is not None and r.get(date_col):
                data.append({"date": str(r[date_col])[:10], "variable": c, "value": v, "site": label(r.get("site", ""))})
    if not data:
        return None
    enc = {"x": {"field": "date", "type": "temporal", "title": None},
           "y": {"field": "value", "type": "quantitative", "title": None}}
    if len(sites) > 1:
        enc["color"] = {"field": "site", "type": "nominal"}
    return {"$schema": VL, "title": title, "data": {"values": data[:MAX_POINTS * 3]},
            "facet": {"row": {"field": "variable", "type": "nominal", "title": None}},
            "resolve": {"scale": {"y": "independent"}},
            "spec": {"mark": {"type": "line", "point": True, "tooltip": True}, "encoding": enc,
                     "width": 520, "height": 90}}


def _bars_per_variable(rows, cols, title, row_label_col=None):
    data = []
    for r in rows[:6]:
        who = label(r.get(row_label_col, "")) if row_label_col else ""
        if r.get("date"):
            who = (who + " " + str(r["date"])[:10]).strip()
        for c in cols:
            v = num(r.get(c))
            if v is not None:
                data.append({"variable": c, "value": v, "row": who or "value"})
    if not data:
        return None
    return _spec(title, data, mark={"type": "bar", "tooltip": True},
                 encoding={"x": {"field": "variable", "type": "nominal", "title": None, "sort": cols},
                           "y": {"field": "value", "type": "quantitative", "title": None},
                           "xOffset": {"field": "row"}, "color": {"field": "row", "type": "nominal", "title": None}})


def _fwi_trend(rows, title):
    data = []
    for r in rows[:6]:
        site = label(r.get("site", ""))
        if num(r.get("previousFWI")) is not None and r.get("previousDate"):
            data.append({"site": site, "date": str(r["previousDate"])[:10], "FWI": num(r["previousFWI"])})
        if num(r.get("FWI")) is not None and r.get("date"):
            data.append({"site": site, "date": str(r["date"])[:10], "FWI": num(r["FWI"])})
    if len(data) < 2:
        return None
    return _spec(title, data, mark={"type": "line", "point": True, "tooltip": True},
                 encoding={"x": {"field": "date", "type": "temporal", "title": None},
                           "y": {"field": "FWI", "type": "quantitative"},
                           "color": {"field": "site", "type": "nominal", "title": None}})


def _observed_vs_threshold(rows, title):
    data = []
    for r in rows[:20]:
        name = label(r.get("ruleLabel") or r.get("condition") or r.get("property"))
        for m, c in (("observed", "observedValue"), ("threshold", "threshold")):
            v = num(r.get(c))
            if v is not None:
                data.append({"condition": f"{name} ({label(r.get('property'))} {r.get('operator', '')})",
                             "measure": m, "value": v})
    if not data:
        return None
    return _spec(title, data, mark={"type": "bar", "tooltip": True},
                 encoding={"y": {"field": "condition", "type": "nominal", "title": None},
                           "x": {"field": "value", "type": "quantitative"},
                           "yOffset": {"field": "measure"},
                           "color": {"field": "measure", "type": "nominal", "title": None,
                                     "scale": {"domain": ["observed", "threshold"], "range": ["#d9480f", "#495057"]}}})


def _generic(cols, rows, title):
    nums = numeric_cols(cols, rows)
    if not nums or not 2 <= len(rows) <= 40:
        return None
    date_col = _date_col(cols, rows)
    if date_col:
        return _time_series(rows, nums[:6], date_col, title)
    cats = [c for c in cols if c not in nums and not ID_LIKE.match(c)]
    if not cats:
        return None
    cat = cats[0]
    data = [{"item": label(r.get(cat)), "variable": c, "value": num(r.get(c))}
            for r in rows for c in nums[:3] if num(r.get(c)) is not None]
    enc = {"x": {"field": "item", "type": "nominal", "title": cat}, "y": {"field": "value", "type": "quantitative",
                                                                           "title": nums[0] if len(nums) == 1 else None}}
    if len(nums) > 1:
        enc["xOffset"] = {"field": "variable"}
        enc["color"] = {"field": "variable", "type": "nominal", "title": None}
    return _spec(title, data, mark={"type": "bar", "tooltip": True}, encoding=enc)


# ------------------------------------------------------------------ entry point
def chart_for(cq_id, columns, rows, title=None):
    if not config.ENABLE_KG_CHARTS or not rows:
        return []
    src = f"knowledge graph ({cq_id})" if cq_id else "knowledge graph (generated SPARQL)"
    t = title or cq_id or "Result"
    charts = []
    try:
        if cq_id == "CQ05":
            charts.append(("Model scores vs. decision thresholds", _scores_vs_threshold(rows, "Model scores vs. thresholds")))
        elif cq_id == "CQ04":
            data_rows = [{**r, "aboveThreshold": "true"} for r in rows]
            charts.append(("Label confidences", _scores_vs_threshold(data_rows, "Label confidences")))
        elif cq_id in ("CQ08", "CQ09"):
            cols = [c for c in (FWI_COLS if cq_id == "CQ09" else WEATHER_COLS) if c in columns]
            date_col = _date_col(columns, rows)
            if date_col:
                charts.append(("Time series", _time_series(rows, cols, date_col,
                                                           "FWI system indices" if cq_id == "CQ09" else "Weather")))
            else:
                charts.append(("Values", _bars_per_variable(rows, cols, "FWI system indices" if cq_id == "CQ09"
                                                            else "Weather", "site")))
            if cq_id == "CQ09":
                charts.append(("FWI trend", _fwi_trend(rows, "FWI: previous record vs. this record")))
        elif cq_id == "CQ10":
            charts.append(("Danger conditions", _observed_vs_threshold(rows, "Observed weather vs. danger-rule thresholds")))
        else:
            charts.append((t, _generic(columns, rows, t)))
    except Exception as e:                       # a chart must never break an answer
        return [{"kind": "note", "title": "chart not available", "text": str(e)[:200]}]
    return [{"kind": "vega", "title": name, "spec": spec, "source": src} for name, spec in charts if spec]
