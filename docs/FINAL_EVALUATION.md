# Final evaluation round

Run this once, after the data and labels are complete, and report only these numbers.
All commands run from the repository root. Everything a step writes goes to `results/eval/`.

## 0. Freeze the setup

```bash
nvidia-smi                                   # GPU visible (driver OK)
ollama ps                                    # after a test call: PROCESSOR = 100% GPU
mkdir -p results/final
ollama list            > results/final/models.txt      # model names + digests
python --version      >> results/final/models.txt
git rev-parse HEAD     > results/final/code_version.txt 2>/dev/null || date > results/final/code_version.txt
```
Do not change prompts, models, labels or code between steps 3 and 8. Routing and answering use temperature 0.

## 1. Complete the knowledge graph

```bash
python scripts/extract_scenes.py --images data/test/images        # all images (resumable)
# audio: run your YAMNet pipeline on data/test/audio -> data/raw/audio_*.json
python scripts/lift.py
python graphdb/load_graphdb.py
python -m qa.evaluate e2e --pool all --systems cq --limit 1         # prints the coverage report
```
Continue only when the coverage report shows (almost) every image, audio and weather question "with data".

## 2. Freeze the CQ labels

Review the most confused CQ pairs in `results/eval/routing_summary.md` (CQ14/CQ15, CQ21, CQ24/CQ25, CQ11/CQ12, CQ07),
fix wrong labels in `docs/FFVQA_CQ_mapping.xlsx`, then rebuild the example index:
```bash
python -m qa.build_index
```

## 3. Ontology evaluation

```bash
python scripts/run_cqs.py                     2>&1 | tee results/final/run_cqs.log
bash validation/validate_ontology.sh          2>&1 | tee results/final/robot.log
python validation/reason_hermit.py            2>&1 | tee results/final/hermit.log
python validation/oops_check.py               2>&1 | tee results/final/oops.log
python -m pytest -q tests                     2>&1 | tee results/final/pytest.log
```

## 4. Routing and CQ selection (all labelled questions, leave-one-out)

```bash
python -m qa.evaluate routing --pool all --limit 5000
```

## 5. End-to-end (same questions for every system)

Run the baselines first, then the main systems: `human_check.csv` is written from the first system of the LAST run.
```bash
python -m qa.evaluate e2e --pool all --limit 5000 --systems text2sparql text2sparql_plus --judge --embed-sim
IMAGES_DIR=data/test/images \
python -m qa.evaluate e2e --pool all --limit 5000 --systems cq_fusion cq cq_vlm vlm --judge --embed-sim
```
Each run ends with error counts. Any `!! … ERROR` line: fix the cause and re-run that command.

Optional ablation of text-to-SPARQL (copy the CSV after each run, it is overwritten):
```bash
for sw in T2S_BIND_CONTEXT T2S_SCHEMA_LINK T2S_ADAPT_TEMPLATE T2S_EXEC_FEEDBACK; do
  cp results/eval/e2e_text2sparql_plus.csv results/final/e2e_text2sparql_plus_full.csv 2>/dev/null
  env $sw=0 python -m qa.evaluate e2e --pool all --limit 100 --systems text2sparql_plus
  cp results/eval/e2e_text2sparql_plus.csv results/final/ablation_without_$sw.csv
done
cp results/final/e2e_text2sparql_plus_full.csv results/eval/e2e_text2sparql_plus.csv
```

## 6. Multimodal RAG (figures)

```bash
python -m qa.build_index --docs-only
python -m qa.evaluate figures --make-template      # fill qa/eval/figure_questions.csv: 30+ questions
python -m qa.evaluate figures --judge
```

## 7. Judge reliability

Fill the `human_grade` column of `results/eval/human_check.csv` (correct / partial / wrong / no_answer) without looking
at `judge_grade` first, then:
```bash
python -m qa.evaluate agreement
```

## 8. Summary tables

```bash
python -m qa.evaluate prf
python -m qa.evaluate compare --systems cq_fusion cq cq_vlm vlm text2sparql text2sparql_plus
```
`final_comparison.md`: every metric with a 95 % CI for every system on the same questions, the yes/no majority
baseline, and exact McNemar tests against the first system.

## 9. Error analysis (sample)

Open `results/eval/e2e_cq_fusion.csv`, take 30 rows with `judge_score` 0 and label each with one cause:

| Cause | How to recognise it |
|---|---|
| extraction | `gold_n_rows` = 0, or `rows_preview` lacks the object |
| CQ choice | `router_cq` ≠ `gold_cq` |
| query | right CQ, `result_F1` low |
| answer generation | rows contain the answer, text is wrong |
| reference / label | the dataset answer is questionable |

## 10. Archive

```bash
cp -r results/eval results/final/eval_$(date +%Y%m%d)
cp config.env results/final/ 2>/dev/null; env | grep -E "^(LLM|SPARQL|EMBED|VISION)_MODEL|^T2S_|^ANSWER_FUSION|^VLM_FALLBACK" >> results/final/models.txt
```
