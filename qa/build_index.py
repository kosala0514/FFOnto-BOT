"""
Build the two embedding indexes used at question time.

    python -m qa.build_index                      # all labelled questions + qa/rag_docs
    python -m qa.build_index --holdout 0.2        # keep 20 % of questions out (for qa.evaluate)
    python -m qa.build_index --gk-answers         # also index the dataset's general-knowledge answers as RAG passages
    python -m qa.build_index --no-figures         # text + tables only (no vision model needed)
    python -m qa.build_index --docs-only          # rebuild only the document index (after adding PDFs)

Run Ollama first:  ollama pull bge-m3          (embedding model)
                   ollama pull qwen2.5vl:7b    (vision model that describes PDF figures; once, results are cached)
PDF figures/tables need:  pip install pymupdf
"""
import argparse
from . import config, rag
from .examples import ExampleIndex, load_from_mapping, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=float, default=0.0, help="fraction of questions kept out for evaluation")
    ap.add_argument("--gk-answers", action="store_true",
                    help="index the ForestFire-VQA general-knowledge answers as RAG passages "
                         "(do NOT use when evaluating RAG on those same questions)")
    ap.add_argument("--no-figures", action="store_true", help="do not extract/describe figures (text and tables only)")
    ap.add_argument("--docs-only", action="store_true", help="skip the labelled-question index")
    ap.add_argument("--vision-model", default=None, help=f"override VISION_MODEL (default {config.VISION_MODEL})")
    a = ap.parse_args()

    if a.docs_only:
        n = rag.build_doc_index(figures=not a.no_figures, vision_model=a.vision_model)
        print(f"RAG index: {n} items")
        return
    examples = load_from_mapping()
    train, test = split(examples, a.holdout) if a.holdout else (examples, [])
    print(f"labelled questions: {len(examples)}  (index: {len(train)}, held out: {len(test)})")
    idx = ExampleIndex.build(train)
    idx.save()
    if test:
        import json
        (config.INDEX_DIR / "heldout.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in test), encoding="utf-8")
    print(f"example index saved to {config.INDEX_DIR}")

    extra = []
    if a.gk_answers:
        extra = [{"text": f"Q: {e['question']}\nA: {e['answer']}", "source": "ForestFire-VQA general knowledge"}
                 for e in train if e["route"] == "rag" and e["answer"]]
        print(f"adding {len(extra)} general-knowledge Q&A pairs as RAG passages")
    n = rag.build_doc_index(extra, figures=not a.no_figures, vision_model=a.vision_model)
    print(f"RAG index: {n} items")


if __name__ == "__main__":
    main()
