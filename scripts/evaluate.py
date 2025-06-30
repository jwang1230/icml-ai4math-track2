#!/usr/bin/env python3
"""
evaluate.py – Dev-set scorer turned to consume extract_prediction.py outputs

Now expects:
  • PRED_PATH: a JSON array of dev-entries with a "prediction" field
  • GT_PATH:   the original dev.json (with ground-truth "answer" & "sig_figs")

Usage
-----
python3 evaluate.py  PRED_PATH  GT_PATH  [--rtol 0.01] [--atol 1e-3]
"""
import argparse, json, math, re, itertools, sys
from decimal import Decimal, ROUND_HALF_UP
import sympy as sp
from sympy.parsing.latex import parse_latex

# ───────────────────────── CLI ─────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("pred_path",
                help="JSON array with full dev entries + 'prediction' field")
ap.add_argument("gt_path", help="Original dev.json with ground‐truth answers")
ap.add_argument("--rtol", type=float, default=0.01,
                help="relative tolerance fallback")
ap.add_argument("--atol", type=float, default=1e-3,
                help="absolute tolerance fallback")
args = ap.parse_args()

# ───────────────────── helpers / cleaning ─────────────────────
DASH_NORM = str.maketrans({"–": "-", "—": "-"})
LATEX_SCI = re.compile(
    r"(?P<coef>[-+]?\d+(?:\.\d+)?)\s*(?:\\times|×|x)\s*10\^\{(?P<exp>[-+]?\d+)\}",
    re.VERBOSE
)
UNICODE_SCI = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*[×✕]\s*10([⁺⁻\d]+)")
SUPERS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
UNIT_RE = re.compile(r"(\\mathrm\{.*?\}|\\text\{.*?\})")

def latex_sci_to_float(s):
    return LATEX_SCI.sub(
        lambda m: str(float(m.group("coef")) * 10**int(m.group("exp"))),
        s
    )

def unicode_sci_to_float(s):
    def _repl(m):
        coef = float(m.group(1))
        exp_txt = m.group(2).translate(SUPERS)
        try:
            exp = int(exp_txt)
        except ValueError:
            # leave unmatched superscript expressions unchanged
            return m.group(0)
        return str(coef * 10**exp)
    return UNICODE_SCI.sub(_repl, s)

def clean(s):
    s = s.translate(DASH_NORM)
    s = latex_sci_to_float(s)
    s = unicode_sci_to_float(s)
    s = UNIT_RE.sub(" ", s)
    s = re.sub(r"^\$|\\\(|\$$|\\\)$", "", s)
    return s.strip()

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
def extract_nums(s):
    return [float(x) for x in NUM_RE.findall(s)]

def round_sig(x, sig):
    if x == 0:
        return "0"
    d = Decimal(str(x))
    exp = -(d.adjusted() - sig + 1)
    q = Decimal(10) ** exp
    r = d.quantize(q, rounding=ROUND_HALF_UP)
    out = format(r, "f")
    return out.rstrip("0").rstrip(".") if "." in out else out

def safe_sigfig(v, default=3):
    try:
        return int(str(v).strip())
    except:
        return default

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
def extract_answer_tag(txt):
    m = ANSWER_RE.search(txt)
    return m.group(1).strip() if m else txt.strip()

# ───────────────────── load GT & Predictions ─────────────────────
try:
    preds = json.load(open(args.pred_path, encoding="utf-8"))
except Exception as e:
    sys.exit(f"[error] loading predictions {args.pred_path}: {e}")
try:
    gt_items = json.load(open(args.gt_path, encoding="utf-8"))
except Exception as e:
    sys.exit(f"[error] loading ground-truth {args.gt_path}: {e}")

gt_map   = {int(d["index"]): str(d["answer"])   for d in gt_items}
sig_map  = {int(d["index"]): safe_sigfig(d.get("sig_figs")) for d in gt_items}
pred_map = {int(d["index"]): d.get("prediction", "")     for d in preds}

# ───────────────────── evaluation loop ─────────────────────
correct, total, mismatches = 0, 0, []

for idx, true_raw in gt_map.items():
    total += 1
    if idx not in pred_map or not pred_map[idx].strip():
        # missing or empty prediction is counted as wrong
        mismatches.append((idx, "<no prediction>", true_raw))
        continue

    raw_pred = extract_answer_tag(pred_map[idx])
    p_txt, t_txt = clean(raw_pred), clean(true_raw)
    p_nums, t_nums = extract_nums(p_txt), extract_nums(t_txt)
    sf = sig_map[idx]

    # 1) sig‐fig match
    if p_nums and t_nums and len(p_nums) == len(t_nums):
        if all(round_sig(a, sf) == round_sig(b, sf)
               for a, b in zip(p_nums, t_nums)):
            correct += 1
            continue

    # 2) numeric tolerance fallback
    if p_nums and t_nums and len(p_nums) == len(t_nums):
        if all(math.isclose(a, b, rel_tol=args.rtol, abs_tol=args.atol)
               for a, b in zip(p_nums, t_nums)):
            correct += 1
            continue

    # 3) symbolic equivalence (no numbers)
    if not p_nums and not t_nums:
        try:
            pe = sp.parse_expr(p_txt, evaluate=False)
            te = parse_latex(t_txt)
            if sp.simplify(pe - te) == 0:
                correct += 1
                continue
        except:
            pass

    # 4) exact cleaned‐string match
    if p_txt.lower() == t_txt.lower():
        correct += 1
    else:
        mismatches.append((idx, raw_pred, true_raw))

# ───────────────────── report ─────────────────────
acc = correct / total if total else 0
print(f"✅ Accuracy: {correct}/{total} = {acc:.2%} "
      f"(sig-fig, rtol={args.rtol}, atol={args.atol})")

if mismatches:
    print("\n❌ Mismatches (first 10):")
    for idx, p, t in itertools.islice(mismatches, 10):
        print(f"[{idx}] Pred: {p!r} | True: {t!r}")
