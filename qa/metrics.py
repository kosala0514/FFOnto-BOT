"""
Evaluation metrics (pure Python, no extra packages needed).

Classification (routing, CQ selection) : accuracy, per-class precision/recall/F1, macro-F1
Query / execution                       : execution rate, result-set precision/recall/F1 (KGQA / QALD style)
Answer text vs reference                : exact match, token F1 (SQuAD), BLEU-1/BLEU-4 (smoothed),
                                          ROUGE-L F1, yes/no agreement, embedding cosine (optional)
Statistics                              : bootstrap 95 % confidence intervals, exact McNemar test
"""
import math
import random
import re
import string
from collections import Counter, defaultdict

_ARTICLES = {"a", "an", "the"}


# ------------------------------------------------------------------ text normalisation
def normalize(text):
    t = str(text or "").lower()
    t = t.replace("—", " ").replace("–", " ")
    t = "".join(ch if ch not in string.punctuation else " " for ch in t)
    return " ".join(w for w in t.split() if w not in _ARTICLES)


def tokens(text):
    return normalize(text).split()


# ------------------------------------------------------------------ answer metrics
def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))


def token_f1(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu(pred, ref, max_n=4):
    """Sentence BLEU with uniform weights, brevity penalty and +1 smoothing for n>1 (Lin & Och, 2004)."""
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        pn, rn = _ngrams(p, n), _ngrams(r, n)
        overlap = sum((pn & rn).values())
        total = max(sum(pn.values()), 0)
        if n == 1:
            if overlap == 0:
                return 0.0
            logs.append(math.log(overlap / total))
        else:
            logs.append(math.log((overlap + 1) / (total + 1)))
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / len(p))
    return bp * math.exp(sum(logs) / max_n)


def rouge_l(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


_NEG = re.compile(r"\b(no|not|none|never|absent|without)\b")


def polarity(text):
    """'yes' / 'no' for yes-no style answers, else None."""
    t = normalize(text)
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no ") or t == "no":
        return "no"
    if re.match(r"^(there (is|are) no|not |none|no evidence|knowledge graph has no)", t):
        return "no"
    return None


def yes_no_agreement(pred, ref):
    """1/0 if the reference is a yes/no answer, else None (not applicable)."""
    pr = polarity(ref)
    if pr is None:
        return None
    pp = polarity(pred) or ("no" if _NEG.search(normalize(pred)) else "yes")
    return float(pp == pr)


def cosine(u, v):
    nu = math.sqrt(sum(x * x for x in u)) or 1e-9
    nv = math.sqrt(sum(x * x for x in v)) or 1e-9
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


# ------------------------------------------------------------------ result sets (execution accuracy)
def _cells(rows, skip_ids=True):
    out = set()
    for r in rows:
        for v in r.values():
            v = str(v).strip()
            if not v or (skip_ids and (v.startswith("ffd:") or v.startswith("http"))):
                continue
            out.add(normalize(v))
    return out


def result_set_prf(pred_rows, gold_rows):
    """Precision/recall/F1 over the (non-identifier) values returned – the usual KGQA answer-set measure."""
    p, g = _cells(pred_rows), _cells(gold_rows)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    tp = len(p & g)
    prec, rec = tp / len(p), tp / len(g)
    f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


# ------------------------------------------------------------------ classification
def classification_report(gold, pred):
    labels = sorted(set(gold) | set(pred))
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1; fn[g] += 1
    per = {}
    for l in labels:
        pr = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        rc = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[l] = {"precision": pr, "recall": rc, "f1": f1, "support": tp[l] + fn[l]}
    gold_labels = [l for l in labels if per[l]["support"] > 0]
    macro = sum(per[l]["f1"] for l in gold_labels) / len(gold_labels) if gold_labels else 0.0
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0
    return {"accuracy": acc, "macro_f1": macro, "per_class": per}


def confusion(gold, pred):
    m = defaultdict(Counter)
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


# ------------------------------------------------------------------ statistics
def bootstrap_ci(values, n=1000, alpha=0.05, seed=0):
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    lo, hi = means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]
    return (sum(vals) / len(vals), lo, hi)


def mcnemar_exact(a_correct, b_correct):
    """Exact (binomial) McNemar test on paired 0/1 outcomes. Returns (b, c, p_value)."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * p)


def cohen_kappa(a, b):
    """Agreement between two raters (e.g. LLM judge vs human) beyond chance."""
    pairs = [(x, y) for x, y in zip(a, b) if x not in (None, "") and y not in (None, "")]
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
