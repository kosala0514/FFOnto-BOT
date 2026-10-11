# Evaluating the FFOnto question-answering system

## Why a correct final answer is not enough

A correct answer can come out of a broken pipeline:

- The router picks the **wrong CQ**, but the LLM still writes "yes" because yes is the most likely answer.
- The query returns **no rows**, and the LLM answers from its own knowledge instead of the graph (a hallucination that happens to be right).
- A wrong answer can also be a **data** problem rather than a system problem: the VLM never extracted the helmet.

So the system is measured **stage by stage**. Each error is attributed to the stage that caused it, and **faithfulness** checks that the answer really comes from the knowledge graph.

```
question ─► routing ─► CQ choice ─► query execution ─► answer text ─► faithful to evidence?
             (1)          (2)            (3)               (4)               (5)
```

## Metrics

| Stage | Metric | Standard / source | Why |
|---|---|---|---|
| 1 Routing (KG vs RAG vs chit-chat) | accuracy, per-class P/R/F1, macro-F1 | classification | Routing is a classification task |
| 2 CQ selection | accuracy, macro-F1, confusion pairs | classification / semantic parsing | "Did it pick the right query template?" |
| 3 Execution | execution rate; result-set P/R/F1 against the gold CQ's rows | KGQA (QALD-style answer-set F1), text-to-SQL execution accuracy | Did the query run and return the right facts? |
| 4 Answer | exact match, token-F1 | SQuAD / VQA | Short factual answers |
| | BLEU-1, BLEU-4 (smoothed) | MT / captioning (Papineni et al. 2002) | Standard, comparable with other papers |
| | ROUGE-L | summarisation (Lin 2004) | Order-aware overlap |
| | yes/no agreement | VQA | About half of your questions are yes/no; polarity is what matters |
| | embedding cosine (bge-m3) | semantic similarity, BERTScore-like | Paraphrases score well |
| | LLM-judge correctness (correct / partial / wrong) + human check with Cohen's κ | LLM-as-a-judge | Meaning-level correctness |
| 5 Faithfulness | supported / partially / unsupported by the returned rows or passages | RAG evaluation (RAGAS-style) | Proves the answer comes from the ontology-based knowledge graph |
| All | latency (s/question) | – | Practicality on a local GPU |
| Comparison | bootstrap 95 % CI; exact McNemar test | statistics | Is CQ mode really better than text-to-SPARQL? |

### About BLEU (and ROUGE)

**Yes, report BLEU**: reviewers expect it, and it makes your numbers comparable with VQA and captioning papers. **Don't make it the main metric**, though. BLEU and ROUGE count shared words, not meaning. In this project's own metric tests:

| System answer vs reference *"Yes—two active flame fronts are visible from this camera."* | token-F1 | BLEU-4 | yes/no |
|---|---|---|---|
| "Yes, two active flame fronts are visible." (correct) | 0.82 | 0.65 | ✔ |
| "No flame fronts are visible." (**opposite meaning**) | 0.53 | 0.28 | ✘ |

The wrong answer still scores 0.53 token-F1. Only yes/no agreement and the judge catch it. Your ForestFire-VQA reference answers are also full sentences written by humans, which penalises correct but differently worded answers. Report BLEU/ROUGE as secondary metrics, and use **yes/no agreement, judge correctness (checked against humans) and faithfulness** as the main ones.

## Procedure

```bash
# 0. keep 20 % of the labelled questions unseen (the router's examples come only from the other 80 %)
python -m qa.build_index --holdout 0.2

# 1. Routing + CQ selection – runs now, no scene data needed
python -m qa.evaluate routing --limit 300
#    -> results/eval/routing_summary.md, routing_predictions.csv

# 2. End-to-end – needs the dataset's images / audio / weather rows in GraphDB
#    first extract scene graphs for the ForestFire-VQA images (also creates their image records):
python scripts/extract_scenes.py --images /path/to/ForestFire-VQA/images
python scripts/lift.py && python graphdb/load_graphdb.py
#    then evaluate; the run starts with a coverage table (results/eval/coverage.csv lists missing sources)
python -m qa.evaluate e2e --systems cq text2sparql --judge --embed-sim --include-rag
#    more questions without leakage: every labelled question, leave-one-out routing
python -m qa.evaluate e2e --pool all --systems cq text2sparql --judge --embed-sim
#    -> results/eval/e2e_summary.md, e2e_cq.csv, e2e_text2sparql.csv, human_check.csv

# 3. Validate the LLM judge: fill human_grade (correct/partial/wrong/no_answer) for the 50 sampled rows
python -m qa.evaluate agreement
#    -> Cohen's kappa; report it next to the judge scores

# 4. Multimodal RAG ablation – after PDFs with figures are indexed (docs/MULTIMODAL_RAG.md)
python -m qa.evaluate figures --make-template   # then write questions for the figures
python -m qa.evaluate figures --judge
#    -> results/eval/figures_summary.md: figure Recall@1/@3, MRR; text-only vs multimodal RAG; figure citation rate
```

## Text-to-SPARQL: baseline and improved version (ablation)

The naive baseline lets the LLM write the whole query; it only *mentions* the selected record, so it often returns facts
about every record or date (many false positives). `text2sparql_plus` adds four mechanisms, each switchable:

| Switch (env var, default 1) | Mechanism |
|---|---|
| `T2S_BIND_CONTEXT` | the code binds `?record` / `?eventCode` with VALUES and filters `?date` rows – the model never writes the IRI |
| `T2S_SCHEMA_LINK` | the prompt lists the vocabulary concepts the question mentions (schema linking) |
| `T2S_ADAPT_TEMPLATE` | the model edits the most similar tested CQ query instead of writing from scratch |
| `T2S_EXEC_FEEDBACK` | one regeneration if the result is empty or larger than `T2S_MAX_ROWS` for a single record/date |

```bash
python -m qa.evaluate e2e --pool all --systems cq text2sparql text2sparql_plus --judge --embed-sim
# ablation: switch one mechanism off, re-run text2sparql_plus, compare result-set F1
T2S_BIND_CONTEXT=0 python -m qa.evaluate e2e --pool all --systems text2sparql_plus
```

## Things to keep honest

1. **Silver labels.** The CQ labels in `FFVQA_CQ_mapping.xlsx` were assigned by keyword rules and then hand-corrected where obviously wrong. Before reporting stage 2, review the confusion pairs in `routing_summary.md`, fix wrong labels in the spreadsheet, rebuild, and re-run. Say in the paper that the labels were reviewed.
2. **No leakage.** The held-out questions never appear as router examples. Don't use `--gk-answers` when evaluating RAG on the dataset's own general-knowledge questions.
3. **Separate perception errors from reasoning errors.** For wrong answers, check whether the scene graph had the fact. If it didn't, the cause is VLM extraction, not the ontology or the query. Report the three error sources separately: ontology gap, VLM extraction, query/rule.
4. **Use the same model and settings for both systems.** CQ mode and text-to-SPARQL must share the LLM and temperature (0 for routing). Report the model name and quantisation.
5. **Report confidence intervals.** With 100–300 questions, differences under a few points are usually not significant. Use the McNemar p-value.

## Paper table templates

**Table A – Routing and CQ selection** (from `routing_summary.md`)

| Model | Routing acc. | Routing macro-F1 | CQ acc. | CQ macro-F1 | s/question |
|---|---|---|---|---|---|
| qwen2.5:14b | | | | | |
| llama3.1:8b | | | | | |

**Table B – End-to-end, CQ mode vs free text-to-SPARQL** (from `e2e_summary.md`)

| Metric | CQ templates | Text-to-SPARQL | McNemar p |
|---|---|---|---|
| Execution rate | | | |
| Result-set F1 | | | – |
| Yes/no agreement | | | |
| Judge correctness | | | |
| Faithfulness | | | – |
| BLEU-4 / ROUGE-L / token-F1 | | | – |

**Table C – Per-CQ results**: from the per-CQ section of `e2e_summary.md`. It shows which question types work and which don't.
