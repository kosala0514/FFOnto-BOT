"""
Evaluation of the FFOnto question-answering system (one module, three commands).

  1) Routing + CQ selection – no knowledge-graph data needed
       python -m qa.build_index --holdout 0.2
       python -m qa.evaluate routing --limit 300

  2) End-to-end – needs real records/scene graphs in GraphDB for the held-out questions
       python -m qa.evaluate e2e --systems cq text2sparql --judge --embed-sim

  5) Precision / recall / F1 for every stage, from the saved CSVs (no model calls)
       python -m qa.evaluate prf                 # -> results/eval/prf_summary.md

  4) Multimodal RAG – figure retrieval and text-only vs multimodal answers (ablation)
       python -m qa.evaluate figures --make-template     # lists every indexed figure -> write questions
       python -m qa.evaluate figures [--judge]           # Recall@1/@3, MRR, citation rate, answer quality

  3) Human check of the LLM judge (inter-rater agreement)
       # fill the 'human_grade' column of results/eval/human_check.csv, then:
       python -m qa.evaluate agreement

Metrics
  routing        accuracy, macro-F1, per-class P/R/F1 (kg / rag / chitchat)
  CQ selection   accuracy, macro-F1, confusion pairs (gold CQ -> predicted CQ)
  execution      execution rate (query ran), result-set P/R/F1 vs the gold CQ's rows
  answer         EM, token-F1, BLEU-1, BLEU-4, ROUGE-L, yes/no agreement,
                 embedding cosine (--embed-sim), LLM-judge correctness (--judge)
  faithfulness   LLM-judge: is every claim supported by the returned rows / passages? (--judge)
  statistics     bootstrap 95 % CI for every mean; exact McNemar test between systems
All outputs go to results/eval/.
"""
import argparse
import csv
import json
import random
import re
import time
from collections import Counter
from . import config, kg, llm, cqs, pipeline
from . import metrics as M

OUT = config.ROOT / "results" / "eval"
GRADE_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "no_answer": 0.0}

JUDGE_SYSTEM = """You grade a system answer against a reference answer for a question about a wildfire image or data.
Return JSON {"grade": "correct" | "partial" | "wrong" | "no_answer", "why": "<short reason>"}.
correct  = same meaning on the key point (wording may differ);
partial  = some key information right, some missing or wrong;
wrong    = contradicts the reference or answers something else;
no_answer = the system says it has no information."""

FAITH_SYSTEM = """You check whether an answer is supported by the evidence (knowledge-graph rows or passages) it was given.
Return JSON {"faithfulness": "supported" | "partially_supported" | "unsupported", "why": "<short reason>"}.
Judge only support by the evidence, not correctness in the real world."""

GRADE_SCHEMA = {"type": "object", "properties": {"grade": {"type": "string", "enum": list(GRADE_SCORE)},
                                                 "why": {"type": "string"}}, "required": ["grade", "why"]}
FAITH_SCHEMA = {"type": "object", "properties": {"faithfulness": {"type": "string",
                "enum": ["supported", "partially_supported", "unsupported"]}, "why": {"type": "string"}},
                "required": ["faithfulness", "why"]}
FAITH_SCORE = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}


def load_heldout():
    f = config.INDEX_DIR / "heldout.jsonl"
    if not f.exists():
        raise SystemExit("run  python -m qa.build_index --holdout 0.2  first")
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def fmt_ci(vals):
    m, lo, hi = M.bootstrap_ci(vals)
    n = sum(v is not None for v in vals)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}] (n={n})" if n else "n/a"


# =================================================================== 1. routing + CQ selection
def stage_routing(test, limit, loo=False):
    rows, t0 = [], time.time()
    for i, e in enumerate(test[:limit]):
        rec_hint = "an image or audio record is selected" if e.get("modality") in ("Image", "Audio") else "none"
        try:
            d = pipeline.route(e["question"], rec_hint, exclude_self=loo)
            pr, pc = d["route"], d["cq"]
        except Exception as ex:
            pr, pc = "error", "error"
            print("  error:", str(ex)[:100])
        rows.append({"question": e["question"], "modality": e.get("modality"),
                     "gold_route": e["route"], "pred_route": pr,
                     "gold_cq": e["cq"] if e["route"] == "kg" else "", "pred_cq": pc if pr == "kg" else "",
                     "route_ok": int(pr == e["route"]),
                     "cq_ok": int(pr == "kg" and pc == e["cq"]) if e["route"] == "kg" else None})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{min(limit, len(test))}")
    write_csv(OUT / "routing_predictions.csv", rows)

    rr = M.classification_report([r["gold_route"] for r in rows], [r["pred_route"] for r in rows])
    kgr = [r for r in rows if r["gold_route"] == "kg"]
    cr = M.classification_report([r["gold_cq"] for r in kgr], [r["pred_cq"] or "(not kg)" for r in kgr])
    conf = Counter((r["gold_cq"], r["pred_cq"] or "(not kg)") for r in kgr if r["gold_cq"] != r["pred_cq"])

    lines = ["# Routing and CQ-selection evaluation\n",
             f"Held-out questions: {len(rows)}  (KG: {len(kgr)}, RAG: {len(rows) - len(kgr)}); "
             f"model {config.LLM_MODEL}; {(time.time() - t0) / max(1, len(rows)):.1f} s/question\n",
             "| Metric | Value (95 % CI) |", "|---|---|",
             f"| Routing accuracy | {fmt_ci([r['route_ok'] for r in rows])} |",
             f"| Routing macro-F1 | {rr['macro_f1']:.3f} |",
             f"| CQ-selection accuracy (KG questions) | {fmt_ci([r['cq_ok'] for r in kgr])} |",
             f"| CQ-selection macro-F1 | {cr['macro_f1']:.3f} |", "",
             "## Routing per class", "| Class | P | R | F1 | support |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
              for k, v in rr["per_class"].items()]
    lines += ["", "## CQ selection per class", "| CQ | P | R | F1 | support |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
              for k, v in sorted(cr["per_class"].items()) if v["support"]]
    lines += ["", "## Most frequent CQ confusions (gold -> predicted)",
              "Check these by hand: some are labelling errors in the silver labels, not model errors.", ""]
    lines += [f"- {g} -> {p}: {c}" for (g, p), c in conf.most_common(15)]
    (OUT / "routing_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\nfull report: {OUT / 'routing_summary.md'}")


# =================================================================== 2. end-to-end
TYPE_OF_MODALITY = {"Image": "ImageRecord", "Audio": "AudioRecord"}


def _stem(x):
    return re.sub(r"\.\w+$", "", str(x or "").strip().rsplit("/", 1)[-1]).lower()


def _record_lookup():
    """(file stem, record type) -> record. Keyed by type so forest_fire_001.jpg and forest_fire_001.wav never collide."""
    by = {}
    for r in kg.list_records():
        for key in {_stem(r["label"]), _stem(r["file"])}:
            if key:
                by[(key, r["type"])] = r
    return by


def _weather_dates():
    q = config.PREFIX_BLOCK + """
SELECT DISTINCT ?d WHERE { ?w a ffonto:WeatherRecord ; ffonto:observationDate ?x . BIND(SUBSTR(STR(?x), 1, 10) AS ?d) }"""
    try:
        return {r["d"] for r in kg.select(q)[1]}
    except Exception:
        return set()


def _source_date(src):
    """Weather questions cite a CSV row ('{"day": 27, "month": 8, "year": 2012, ...') -> '2012-08-27'."""
    g = {k: re.search(rf'"{k}"\s*:\s*(\d+)', str(src)) for k in ("day", "month", "year")}
    if all(g.values()):
        return f"{int(g['year'].group(1)):04d}-{int(g['month'].group(1)):02d}-{int(g['day'].group(1)):02d}"
    return None


def match_context(e, lookup, dates):
    """Return (record, date) that gives this question its data context, or (None, None) if the KG has no data for it."""
    mod = e.get("modality") or ""
    if mod in TYPE_OF_MODALITY:
        rec = lookup.get((_stem(e.get("source")), TYPE_OF_MODALITY[mod]))
        return (rec, None) if rec else (None, None)
    if mod.startswith("Tabular"):
        d = _source_date(e.get("source"))
        return (None, d) if d and d in dates else (None, None)
    return None, None


def coverage_report(pool, lookup, dates):
    """How many KG questions have data in GraphDB, and which sources to process next."""
    from collections import defaultdict
    per_src = defaultdict(lambda: {"n": 0, "modality": "", "in_kg": 0})
    for e in pool:
        if e["route"] != "kg":
            continue
        rec, d = match_context(e, lookup, dates)
        key = str(e.get("source") or "")[:80]
        per_src[key]["n"] += 1
        per_src[key]["modality"] = e.get("modality")
        per_src[key]["in_kg"] = int(bool(rec or d))
    rows = sorted(({"source": k, **v} for k, v in per_src.items()), key=lambda r: (r["in_kg"], -r["n"]))
    write_csv(OUT / "coverage.csv", rows)
    by_mod = Counter()
    for r in rows:
        by_mod[(r["modality"], r["in_kg"])] += r["n"]
    print("\nCoverage of KG questions by data in GraphDB:")
    for mod in sorted({m for m, _ in by_mod}):
        have, miss = by_mod[(mod, 1)], by_mod[(mod, 0)]
        print(f"  {mod:<24} {have:>4} with data, {miss:>4} without")
    missing = [r for r in rows if not r["in_kg"]][:10]
    if missing:
        print("  sources missing from the KG with the most questions (full list: results/eval/coverage.csv):")
        for r in missing:
            print(f"    {r['n']:>3} questions  {r['modality']:<10} {r['source']}")
    print()


def _evidence_text(res):
    parts = []
    if res.get("rows"):
        cols = res.get("columns") or list(res["rows"][0])
        parts.append("\n".join(" | ".join(str(r.get(c, ""))[:60] for c in cols) for r in res["rows"][:30]))
    if res.get("sources"):                     # RAG answers, and the background used by a fused KG answer
        parts.append("\n".join(s.get("text", "")[:300] for s in res["sources"][:8]))
    return "\n\n".join(parts) or "(no evidence)"


def score_answer(e, res, gold_rows, judge, embed_sim):
    ref, pred = e["answer"], res["answer"]
    row = {"question": e["question"], "source": e.get("source"), "gold_route": e["route"],
           "pred_route": res.get("route"), "gold_cq": e["cq"] if e["route"] == "kg" else "",
           "router_cq": res.get("router_cq") or res.get("cq"), "mode": res.get("mode"),
           "route_ok": int(res.get("route") == e["route"]),
           "cq_ok": (int((res.get("router_cq") or res.get("cq")) == e["cq"]) if e["route"] == "kg" else None),
           "exec_ok": (int(res.get("mode") in ("cq", "text2sparql") and not res.get("error"))
                       if e["route"] == "kg" else None),
           "n_rows": len(res.get("rows") or []), "gold_n_rows": (len(gold_rows) if gold_rows is not None else None),
           "fused": int(bool(res.get("fused"))), "seconds": res.get("seconds"),
           "error": (res.get("error") or "")[:300],
           "reference": ref, "answer": pred,
           "EM": M.exact_match(pred, ref), "token_F1": M.token_f1(pred, ref),
           "token_P": M.token_prf(pred, ref)[0], "token_R": M.token_prf(pred, ref)[1],
           "BLEU1": M.bleu(pred, ref, 1), "BLEU4": M.bleu(pred, ref, 4), "ROUGE_L": M.rouge_l(pred, ref),
           "yes_no": M.yes_no_agreement(pred, ref),
           "abstain": (int(M.answer_polarity(pred) == "abstain") if e["route"] == "kg" else None)}
    if e["route"] == "kg" and gold_rows is not None and res.get("mode") in ("cq", "text2sparql"):
        p, r, f = M.result_set_prf(res.get("rows") or [], gold_rows)
        tp, fp, fn = M.result_set_counts(res.get("rows") or [], gold_rows)
        row.update(result_TP=tp, result_FP=fp, result_FN=fn)
        row.update(result_P=p, result_R=r, result_F1=f)
    if embed_sim:
        try:
            a, b = llm.embed([pred, ref])
            row["emb_cosine"] = M.cosine(a, b)
        except Exception:
            row["emb_cosine"] = None
    if judge:
        try:
            j = llm.chat_json(JUDGE_SYSTEM, f"Question: {e['question']}\nReference: {ref}\nSystem answer: {pred}", GRADE_SCHEMA)
            row.update(judge_grade=j["grade"], judge_score=GRADE_SCORE[j["grade"]], judge_why=j["why"])
        except Exception as ex:
            row.update(judge_grade="error", judge_score=None, judge_why=str(ex)[:80])
        try:
            f = llm.chat_json(FAITH_SYSTEM, f"Question: {e['question']}\nEvidence:\n{_evidence_text(res)}\n\nAnswer: {pred}",
                              FAITH_SCHEMA)
            row.update(faithfulness=f["faithfulness"], faith_score=FAITH_SCORE[f["faithfulness"]])
        except Exception:
            row.update(faithfulness="error", faith_score=None)
    return row


def stage_e2e(test, limit, systems, judge, embed_sim, include_rag, loo=False):
    lookup, dates = _record_lookup(), _weather_dates()
    coverage_report(test, lookup, dates)
    chosen = []
    for e in test:
        rec, d = match_context(e, lookup, dates)
        if e["route"] == "kg" and (rec or d):
            chosen.append((e, rec, d))
        elif e["route"] == "rag" and include_rag:
            chosen.append((e, None, None))
    chosen = chosen[:limit]
    if not chosen:
        raise SystemExit("No question in this pool has its image, audio or weather row in GraphDB yet.\n"
                         "Run scripts/extract_scenes.py on the dataset images (and your audio pipeline on the clips), "
                         "then lift.py and load_graphdb.py. Or add --include-rag to evaluate general-knowledge questions only.")
    print(f"evaluating {len(chosen)} questions × systems {systems}")

    results = {s: [] for s in systems}
    for i, (e, rec, day) in enumerate(chosen):
        gold_rows = None
        if e["route"] == "kg" and e["cq"] in cqs.REGISTRY:
            try:
                gold_rows = cqs.run_cq(e["cq"], record=rec["iri"] if rec else None, date=day)["rows"]
            except Exception:
                gold_rows = None
        for s in systems:
            if s == "vlm":
                res = pipeline.vlm_baseline(e["question"], rec["iri"] if rec else None)
            else:
                res = pipeline.answer(e["question"], record=rec["iri"] if rec else None,
                                      record_desc=rec["label"] if rec else (f"weather record of {day}" if day else None),
                                      force_text2sparql=s.startswith("text2sparql"), exclude_self=loo, default_date=day,
                                      t2s_mode="plus" if s == "text2sparql_plus" else "naive",
                                      fusion=s in ("cq_fusion", "cq_vlm"), vlm_fallback=(s == "cq_vlm"))
            row = score_answer(e, res, gold_rows, judge, embed_sim)
            row["record_type"] = rec["type"] if rec else ("WeatherRecord" if day else "")
            row["vlm_used"] = int(bool(res.get("vlm_used")))
            row["rows_preview"] = " || ".join(" | ".join(f"{k}={str(v)[:40]}" for k, v in r.items()
                                                         if not str(v).startswith(("ffd:", "http")))
                                              for r in (res.get("rows") or [])[:6])[:600]
            results[s].append(row)
        print(f"  {i + 1}/{len(chosen)} {e['question'][:60]}")

    for s in systems:
        write_csv(OUT / f"e2e_{s}.csv", results[s])
    for s in systems:
        errs = [r["error"] for r in results[s] if r.get("error")]
        if errs:
            print(f"\n!! {s}: {len(errs)}/{len(results[s])} questions ended in an ERROR. Most frequent:")
            for msg, c in Counter(e[:160] for e in errs).most_common(3):
                print(f"   {c:>4} x {msg}")
    if all(sum(1 for r in results[s] if r.get("error")) > 0.5 * len(results[s]) for s in systems):
        raise SystemExit("\nMost answers failed, so the metrics would be meaningless and were not summarised.\n"
                         "Fix the error above (usually GraphDB or Ollama not running / out of GPU memory), then re-run.\n"
                         "Check quickly with:  curl -s localhost:8000/health   or   "
                         "python -c \"from qa import pipeline; print(pipeline.answer('Is smoke visible?'))\"")
    metrics = ["route_ok", "cq_ok", "exec_ok", "result_P", "result_R", "result_F1", "EM", "token_F1", "BLEU1",
               "BLEU4", "ROUGE_L", "yes_no", "abstain", "emb_cosine", "judge_score", "faith_score", "seconds"]
    lines = ["# End-to-end evaluation\n", f"Questions: {len(chosen)}; model {config.LLM_MODEL}\n",
             "| Metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for m in metrics:
        if any(m in r for s in systems for r in results[s]):
            lines.append(f"| {m} | " + " | ".join(fmt_ci([r.get(m) for r in results[s]]) for s in systems) + " |")
    if len(systems) == 2:
        a, b = systems
        lines += ["", f"## Paired comparison {a} vs {b} (exact McNemar)"]
        for m in ("exec_ok", "yes_no"):
            pa = [r.get(m) for r in results[a]]; pb = [r.get(m) for r in results[b]]
            pairs = [(x, y) for x, y in zip(pa, pb) if x is not None and y is not None]
            if pairs:
                bb, cc, p = M.mcnemar_exact([x for x, _ in pairs], [y for _, y in pairs])
                lines.append(f"- {m}: {a} only correct = {bb}, {b} only correct = {cc}, p = {p:.4f}")
        if judge:
            pa = [r.get("judge_score") == 1.0 for r in results[a]]; pb = [r.get("judge_score") == 1.0 for r in results[b]]
            bb, cc, p = M.mcnemar_exact(pa, pb)
            lines.append(f"- judge correct: {a} only = {bb}, {b} only = {cc}, p = {p:.4f}")
    # per-CQ breakdown for the first system
    s0 = systems[0]
    by_cq = {}
    for r in results[s0]:
        if r["gold_cq"]:
            by_cq.setdefault(r["gold_cq"], []).append(r)
    key = "judge_score" if judge else "token_F1"
    lines += ["", f"## Per-CQ results ({s0}, metric = {key})", "| CQ | n | CQ choice | exec | result F1 | " + key + " |",
              "|---|---|---|---|---|---|"]
    for c, rs in sorted(by_cq.items()):
        mean = lambda k: (lambda v: f"{sum(v) / len(v):.2f}" if v else "–")([x[k] for x in rs if x.get(k) is not None])
        lines.append(f"| {c} | {len(rs)} | {mean('cq_ok')} | {mean('exec_ok')} | {mean('result_F1')} | {mean(key)} |")
    (OUT / "e2e_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    if judge:                                         # sample for the human check
        sample = random.Random(1).sample(results[s0], min(50, len(results[s0])))
        write_csv(OUT / "human_check.csv", [{"question": r["question"], "reference": r["reference"], "answer": r["answer"],
                                             "judge_grade": r.get("judge_grade"), "human_grade": ""} for r in sample])
        print(f"\nhuman check sample: {OUT / 'human_check.csv'} (fill human_grade, then: python -m qa.evaluate agreement)")


# =================================================================== 4. multimodal RAG (figures)
FIG_QS = config.ROOT / "qa" / "eval" / "figure_questions.csv"


def stage_figures(judge, make_template):
    """Gold = the figure (document + page) that answers the question; optional reference answer.
    Reports figure retrieval Recall@1/@3 and MRR, and – if answers are given – compares text-only RAG
    with multimodal RAG on the same questions (token-F1, judge score, gold-figure citation rate)."""
    from . import rag
    if make_template:
        items, _ = rag._load_index()
        figs = [i for i in items if i.get("type") == "figure"]
        if not figs:
            raise SystemExit("No figures in the RAG index. Put PDFs in qa/rag_docs and run: python -m qa.build_index --docs-only")
        write_csv(FIG_QS, [{"doc": f["doc"], "page": f["page"], "file": f["file"], "figure_type": f.get("figure_type"),
                            "caption": f.get("caption", "")[:120], "question": "", "answer": ""} for f in figs])
        print(f"template: {FIG_QS}\nWrite 1-3 questions per useful figure (question + short reference answer), "
              "delete rows you do not use, then run: python -m qa.evaluate figures --judge")
        return
    if not FIG_QS.exists():
        raise SystemExit("run  python -m qa.evaluate figures --make-template  first")
    qs = [r for r in csv.DictReader(FIG_QS.open(encoding="utf-8")) if r.get("question", "").strip()]
    if not qs:
        raise SystemExit(f"{FIG_QS} has no questions yet")
    rows = []
    for i, q in enumerate(qs):
        gold = (q["doc"], str(q["page"]))
        _, figs = rag.retrieve_mm(q["question"], k_fig=5, min_fig_score=0.0)
        ranks = [j + 1 for j, f in enumerate(figs) if (f["doc"], str(f["page"])) == gold]
        r = ranks[0] if ranks else None
        row = {"question": q["question"], "gold": f"{gold[0]} p.{gold[1]}", "rank": r,
               "R@1": float(r == 1), "R@3": float(bool(r) and r <= 3), "RR": 1.0 / r if r else 0.0,
               "top1": figs[0]["source"] if figs else "", "top1_score": round(figs[0]["score"], 3) if figs else None}
        if q.get("answer", "").strip():
            for mode, use in (("text", False), ("mm", True)):
                ans, _, _, used = pipeline.rag_answer(q["question"], use_figures=use)
                row[f"{mode}_answer"] = ans
                row[f"{mode}_token_F1"] = M.token_f1(ans, q["answer"])
                if use:
                    row["mm_cites_gold"] = float(any(f.get("cited") and (f["doc"], str(f["page"])) == gold for f in used))
                    row["mm_cites_any"] = float(any(f.get("cited") for f in used))
                if judge:
                    try:
                        j = llm.chat_json(JUDGE_SYSTEM, f"Question: {q['question']}\nReference: {q['answer']}\n"
                                                        f"System answer: {ans}", GRADE_SCHEMA)
                        row[f"{mode}_judge"] = GRADE_SCORE[j["grade"]]
                    except Exception:
                        row[f"{mode}_judge"] = None
        rows.append(row)
        print(f"  {i + 1}/{len(qs)} rank={r} {q['question'][:60]}")
    write_csv(OUT / "figures.csv", rows)
    lines = ["# Multimodal RAG evaluation\n", f"Questions: {len(rows)}; vision model {config.VISION_MODEL}; "
             f"embeddings {config.EMBED_MODEL}\n", "## Figure retrieval", "| Metric | value |", "|---|---|"]
    for m in ("R@1", "R@3", "RR"):
        lines.append(f"| {'MRR' if m == 'RR' else 'Recall' + m[1:]} | {fmt_ci([r[m] for r in rows])} |")
    if any("mm_answer" in r for r in rows):
        lines += ["", "## Answers: text-only RAG vs multimodal RAG", "| Metric | text-only | multimodal |", "|---|---|---|"]
        for m in ("token_F1", "judge"):
            if any(f"mm_{m}" in r for r in rows):
                lines.append(f"| {m} | {fmt_ci([r.get('text_' + m) for r in rows])} | {fmt_ci([r.get('mm_' + m) for r in rows])} |")
        lines.append(f"| cites the gold figure | – | {fmt_ci([r.get('mm_cites_gold') for r in rows])} |")
        lines.append(f"| cites any figure | – | {fmt_ci([r.get('mm_cites_any') for r in rows])} |")
        if judge:
            pa = [r.get("mm_judge") == 1.0 for r in rows if "mm_judge" in r]
            pb = [r.get("text_judge") == 1.0 for r in rows if "mm_judge" in r]
            bb, cc, pv = M.mcnemar_exact(pa, pb)
            lines.append(f"\nMcNemar (judge correct): multimodal only = {bb}, text-only only = {cc}, p = {pv:.4f}")
    (OUT / "figures_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# =================================================================== precision / recall / F1 report (from saved CSVs)
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stage_prf():
    """Precision, recall and F1 at every stage, recomputed from the saved CSVs (no model calls).
      routing        per class (kg / rag / chitchat), macro and micro
      CQ selection   per CQ, macro and micro (micro = accuracy for single-label choices)
      retrieval      result-set P/R/F1: macro (mean over questions) and micro (pooled TP/FP/FN)
      answer         token-level P/R/F1 against the reference answer
      yes/no         P/R/F1 for the classes "yes" and "no", plus macro-F1"""
    lines = ["# Precision, recall and F1\n"]

    def cls_block(title, gold, pred, labels=None):
        rep = M.classification_report(gold, pred)
        per = {k: v for k, v in rep["per_class"].items() if v["support"] and (labels is None or k in labels)}
        mp = sum(v["precision"] for v in per.values()) / max(1, len(per))
        mr = sum(v["recall"] for v in per.values()) / max(1, len(per))
        out = [f"## {title}", f"n = {len(gold)}", "", "| Class | Precision | Recall | F1 | support |", "|---|---|---|---|---|"]
        out += [f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |" for k, v in sorted(per.items())]
        out += [f"| **macro** | {mp:.3f} | {mr:.3f} | {rep['macro_f1']:.3f} | {len(gold)} |",
                f"| **micro** (= accuracy) | {rep['accuracy']:.3f} | {rep['accuracy']:.3f} | {rep['accuracy']:.3f} | {len(gold)} |", ""]
        return out

    f = OUT / "routing_predictions.csv"
    if f.exists():
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        lines += cls_block("Routing (kg / rag / chitchat)", [r["gold_route"] for r in rows], [r["pred_route"] for r in rows])
        kg_rows = [r for r in rows if r["gold_route"] == "kg"]
        lines += cls_block("CQ selection (knowledge-graph questions)", [r["gold_cq"] for r in kg_rows],
                           [r["pred_cq"] or "(not kg)" for r in kg_rows])

    for sysname in ("cq", "cq_fusion", "cq_vlm", "vlm", "text2sparql", "text2sparql_plus"):
        f = OUT / f"e2e_{sysname}.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        lines += [f"## End-to-end: {sysname}", f"n = {len(rows)} questions", "",
                  "| Stage | Precision | Recall | F1 | n |", "|---|---|---|---|---|"]
        kg = [r for r in rows if _f(r.get("result_F1")) is not None]
        if kg:
            mean = lambda k: sum(_f(r[k]) for r in kg) / len(kg)
            lines.append(f"| Retrieval, macro (mean per question) | {mean('result_P'):.3f} | {mean('result_R'):.3f} | {mean('result_F1'):.3f} | {len(kg)} |")
            if all(_f(r.get("result_TP")) is not None for r in kg):
                tp, fp, fn = (sum(int(float(r[k])) for r in kg) for k in ("result_TP", "result_FP", "result_FN"))
                p, rc, f1 = M.prf_from_counts(tp, fp, fn)
                lines.append(f"| Retrieval, micro (pooled TP={tp}, FP={fp}, FN={fn}) | {p:.3f} | {rc:.3f} | {f1:.3f} | {len(kg)} |")
        tok = [M.token_prf(r["answer"], r["reference"]) for r in rows]
        if tok:
            lines.append(f"| Answer tokens vs reference (macro) | {sum(t[0] for t in tok) / len(tok):.3f} | "
                         f"{sum(t[1] for t in tok) / len(tok):.3f} | {sum(t[2] for t in tok) / len(tok):.3f} | {len(tok)} |")
        def pred_pol(r):
            return "error" if r.get("error") else M.answer_polarity(r["answer"])
        n_err = sum(1 for r in rows if pred_pol(r) == "error")
        if n_err:
            msgs = Counter((r.get("error") or r["answer"])[:70] for r in rows if pred_pol(r) == "error")
            lines += [f"**{n_err} of {len(rows)} answers ended in an error** (a failed generated query is an error too). "
                      "Most frequent: " + "; ".join(f"{c} x {m}" for m, c in msgs.most_common(2)), ""]
        lines.append("")
        yn_rows = [r for r in rows if M.polarity(r["reference"]) in ("yes", "no")]
        if yn_rows:
            out = Counter(pred_pol(r) for r in yn_rows)
            gold_c = Counter(M.polarity(r["reference"]) for r in yn_rows)
            maj, maj_n = gold_c.most_common(1)[0]
            lines += [f"Yes/no answers ({sysname}): " + ", ".join(f"{k} {v}" for k, v in out.most_common())
                      + "  (abstain = 'the knowledge graph has no information …')",
                      f"Reference answers: " + ", ".join(f"{k} {v}" for k, v in gold_c.most_common())
                      + f". Majority baseline (always '{maj}'): accuracy {maj_n / len(yn_rows):.3f}", ""]
            lines += cls_block(f"Yes/no questions ({sysname})", [M.polarity(r["reference"]) for r in yn_rows],
                               [pred_pol(r) for r in yn_rows], labels={"yes", "no"})
            # Separate QA errors from missing data: questions whose gold CQ query found evidence in the KG
            have = [r for r in yn_rows if _f(r.get("gold_n_rows")) not in (None, 0.0)]
            if have and len(have) < len(yn_rows):
                lines += cls_block(f"Yes/no questions where the KG holds evidence ({sysname}, gold query returned rows)",
                                   [M.polarity(r["reference"]) for r in have], [pred_pol(r) for r in have],
                                   labels={"yes", "no"})
                lines += [f"{len(yn_rows) - len(have)} yes/no questions have no evidence in the KG at all "
                          "(the gold query returns 0 rows: the scene/caption/audio extraction did not record it).", ""]
            img = [r for r in yn_rows if r.get("record_type") == "ImageRecord"]
            if img and len(img) < len(yn_rows):                 # same subset for every system, incl. the vlm baseline
                lines += cls_block(f"Yes/no questions about images ({sysname})", [M.polarity(r["reference"]) for r in img],
                                   [pred_pol(r) for r in img], labels={"yes", "no"})
            if any(r.get("vlm_used") == "1" for r in yn_rows):
                used = [r for r in yn_rows if r.get("vlm_used") == "1"]
                ok = sum(1 for r in used if M.polarity(r["reference"]) == "yes")
                lines += [f"Vision-model check changed {len(used)} answers to 'yes'; {ok} of them are 'yes' in the reference.", ""]
    (OUT / "prf_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# =================================================================== 3. judge vs human
def stage_agreement():
    f = OUT / "human_check.csv"
    if not f.exists():
        raise SystemExit(
            "results/eval/human_check.csv does not exist yet.\n"
            "The 'agreement' step is the LAST step. Order:\n"
            "  1. python -m qa.evaluate routing --limit 300\n"
            "  2. python -m qa.evaluate e2e --judge ...      (creates human_check.csv)\n"
            "  3. fill the human_grade column (correct/partial/wrong/no_answer) by hand\n"
            "  4. python -m qa.evaluate agreement")
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    if not any(r.get("human_grade", "").strip() for r in rows):
        raise SystemExit("human_check.csv has no human grades yet – fill the human_grade column first.")
    a = [r["judge_grade"] for r in rows]; b = [r["human_grade"].strip().lower() for r in rows]
    k = M.cohen_kappa(a, b)
    acc = sum(x == y for x, y in zip(a, b) if y) / max(1, sum(1 for y in b if y))
    print(f"LLM judge vs human: raw agreement {acc:.3f}, Cohen's kappa {k:.3f}  (n={sum(1 for y in b if y)})")
    # binary view: is the answer acceptable (correct) or not?  Partial-vs-correct is where raters disagree most.
    bin_ = lambda g: "ok" if g == "correct" else ("-" if not g else "not_ok")
    pairs = [(bin_(x), bin_(y)) for x, y in zip(a, b) if y]
    kb = M.cohen_kappa([x for x, _ in pairs], [y for _, y in pairs])
    print(f"binary (correct vs not): agreement {sum(x == y for x, y in pairs) / len(pairs):.3f}, kappa {kb:.3f}")
    conf = Counter((x, y) for x, y in zip(a, b) if y)
    labels = ["correct", "partial", "wrong", "no_answer"]
    print("\nconfusion (rows = judge, columns = human)")
    print(f"{'':>10} " + " ".join(f"{l:>9}" for l in labels))
    for j in labels:
        print(f"{j:>10} " + " ".join(f"{conf[(j, h)]:>9}" for h in labels))
    print("\nInterpretation (Landis & Koch): <0.2 slight, 0.21-0.40 fair, 0.41-0.60 moderate, "
          "0.61-0.80 substantial, >0.80 almost perfect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["routing", "e2e", "agreement", "figures", "prf"])
    ap.add_argument("--make-template", action="store_true", help="figures: write qa/eval/figure_questions.csv to fill in")
    ap.add_argument("--pool", choices=["heldout", "all"], default="heldout",
                    help="heldout = the 20 %% kept out by build_index --holdout; all = every labelled question, "
                         "leave-one-out (the question itself is never shown to the router as an example)")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--systems", nargs="+", default=["cq", "text2sparql"],
                    choices=["cq", "cq_fusion", "cq_vlm", "vlm", "text2sparql", "text2sparql_plus"],
                    help="text2sparql = naive baseline; text2sparql_plus = with context binding, schema linking, "
                         "template adaptation and execution feedback; cq_fusion = CQ templates, and when the KG-only "
                         "answer abstains, a second answer that interprets the rows with ontology/document background; "
                         "vlm = baseline without the KG (vision model answers from the image; image questions only); "
                         "cq_vlm = cq_fusion + the vision model checks the image when the KG has no positive evidence")
    ap.add_argument("--judge", action="store_true", help="LLM-judge correctness + faithfulness")
    ap.add_argument("--embed-sim", action="store_true", help="embedding cosine between answer and reference")
    ap.add_argument("--include-rag", action="store_true", help="also evaluate general-knowledge (RAG) questions")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "agreement":
        return stage_agreement()
    if a.stage == "prf":
        return stage_prf()
    if a.stage == "figures":
        return stage_figures(a.judge, a.make_template)
    loo = a.pool == "all"
    if loo:
        from .examples import load_from_mapping
        test = load_from_mapping()
        random.Random(7).shuffle(test)             # mix modalities and CQs when --limit is used
    else:
        test = load_heldout()
    print(f"question pool: {a.pool} ({len(test)} questions)")
    if a.stage == "routing":
        stage_routing(test, a.limit, loo)
    else:
        stage_e2e(test, a.limit, a.systems, a.judge, a.embed_sim, a.include_rag, loo)


if __name__ == "__main__":
    main()
