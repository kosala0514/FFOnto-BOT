"""
Labelled example questions (from docs/FFVQA_CQ_mapping.xlsx) and a small embedding index.

At question time the router is shown the k most similar labelled questions
("Is a Pulaski visible?" -> CQ18), which makes CQ selection with a local 7-14B model reliable.
"""
import json
import random
import numpy as np
from . import config, llm

EX_FILE = config.INDEX_DIR / "examples.jsonl"
EX_VEC = config.INDEX_DIR / "examples.npy"


def load_from_mapping(xlsx=None):
    """Read the 'Question Mapping' sheet: question, CQ id, sheet, modality, answer."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx or config.MAPPING_XLSX, read_only=True, data_only=True)
    ws = wb["Question Mapping"]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    idx = {h: i for i, h in enumerate(head)}
    out, seen = [], set()
    for r in rows[1:]:
        q, cq = r[idx["Question"]], r[idx["CQ"]]
        if not q or not cq:
            continue
        q = str(q).strip()
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": q, "cq": str(cq).strip(),
                    "route": "rag" if str(cq).strip() == "OOS" else "kg",
                    "modality": r[idx["Modality"]], "answer": r[idx["Answer (original)"]] or "",
                    "source": str(r[idx["Source"]] or "").strip()})
    return out


def split(examples, test_fraction=0.2, seed=42):
    ex = examples[:]
    random.Random(seed).shuffle(ex)
    n = int(len(ex) * test_fraction)
    return ex[n:], ex[:n]          # train, test


class ExampleIndex:
    def __init__(self, examples, vectors):
        self.examples = examples
        v = np.asarray(vectors, dtype=np.float32)
        self.vectors = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    @classmethod
    def build(cls, examples):
        return cls(examples, llm.embed([e["question"] for e in examples]))

    def save(self):
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        EX_FILE.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in self.examples), encoding="utf-8")
        np.save(EX_VEC, self.vectors)

    @classmethod
    def load(cls):
        if not EX_FILE.exists():
            return None
        ex = [json.loads(l) for l in EX_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls(ex, np.load(EX_VEC))

    def nearest(self, question, k=None, exclude_text=None):
        k = k or config.N_EXAMPLES
        q = np.asarray(llm.embed(question)[0], dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        scores = self.vectors @ q
        order = np.argsort(-scores)
        out = []
        for i in order:
            e = self.examples[int(i)]
            if exclude_text and e["question"].strip().lower() == exclude_text.strip().lower():
                continue
            out.append({**e, "score": float(scores[int(i)])})
            if len(out) >= k:
                break
        return out
