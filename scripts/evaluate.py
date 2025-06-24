#!/usr/bin/env python3
"""
evaluate.py – SeePhys dev-set scorer with proper significant-figure handling.

Usage
-----
python3 evaluate.py  PRED_FILE  GT_FILE  [--rtol 0.01] [--atol 1e-3]

• PRED_FILE – your model outputs (JSON array  *or*  JSONL)
              [{"index": 537, "answer": "-6.5"}, …]
• GT_FILE   – organiser dev.json (or a subset JSON/JSONL)
              must contain "sig_figs" field per item.

Default tolerances: 1 % relative, 1e-3 absolute (only used if sig-fig match fails).
"""

import argparse, json, math, pathlib, re, itertools
from decimal import Decimal, ROUND_HALF_UP

# ----------------------------- CLI ---------------------------------------
ap = argparse.ArgumentParser(description="Evaluate predictions with sig-fig logic")
ap.add_argument("pred_path")
ap.add_argument("gt_path")
ap.add_argument("--rtol", type=float, default=0.01, help="relative tol (default 1%)")
ap.add_argument("--atol", type=float, default=1e-3, help="absolute tol (default 1e-3)")
args = ap.parse_args()

# ----------------------------- helpers -----------------------------------
DASH_NORM = str.maketrans({"–": "-", "—": "-"})

LATEX_SCI_RE = re.compile(
    r"(?P<coef>[-+]?\d+(?:\.\d+)?)\s*(?:\\times|×|x)\s*10\^\{(?P<exp>[-+]?\d+)\}",
    re.VERBOSE,
)

def latex_sci_to_float(s: str) -> str:
    """Convert '4.48 \\times 10^{-3}' → '0.00448' inside a larger string."""
    def _repl(m):
        return str(float(m.group("coef")) * 10 ** int(m.group("exp")))
    return LATEX_SCI_RE.sub(_repl, s)

# strip LaTeX units like \mathrm{T} or \text{m/s}
UNIT_RE = re.compile(r"(\\mathrm\{.*?\}|\\text\{.*?\})")

def clean(txt: str) -> str:
    txt = txt.translate(DASH_NORM).strip()
    txt = latex_sci_to_float(txt)
    txt = UNIT_RE.sub(" ", txt)
    # remove outer $…$, \(…\)
    txt = re.sub(r"^\$|\\\(|\$$|\\\)$", "", txt)
    return txt

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

def extract_nums(s: str):
    return [float(x) for x in NUM_RE.findall(s)]

def round_sig(x: float, sig: int) -> str:
    """Round x to 'sig' significant figures and return canonical string."""
    if x == 0:
        return "0"
    d = Decimal(str(x))
    exponent = -(d.adjusted() - sig + 1)
    q = Decimal(10) ** exponent
    rounded = d.quantize(q, rounding=ROUND_HALF_UP)
    s = format(rounded, "f")
    # strip trailing zeros & dot
    return s.rstrip("0").rstrip(".") if "." in s else s

DEFAULT_SF = 3  # fallback if sig_figs missing/blank

def safe_sigfig(val, default=DEFAULT_SF):
    try:
        return int(str(val).strip())
    except (TypeError, ValueError):
        return default

# ----------------------------- I/O ---------------------------------------
def load(path):
    path = pathlib.Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return json.load(open(path, encoding="utf-8"))

pred_items = load(args.pred_path)
gt_items   = load(args.gt_path)

pred = {int(d["index"]): str(d["answer"]).strip() for d in pred_items}
gt, sigfig = {}, {}
for d in gt_items:
    idx = int(d["index"])
    gt[idx] = str(d["answer"]).strip()
    sigfig[idx] = safe_sigfig(d.get("sig_figs"))

# ----------------------------- evaluate ----------------------------------
correct, total, mism = 0, 0, []

for idx, true_raw in gt.items():
    if idx not in pred:
        continue
    total += 1
    pred_raw = pred[idx]

    p_nums = extract_nums(clean(pred_raw))
    t_nums = extract_nums(clean(true_raw))
    sf     = sigfig[idx]

    # ---- sig-fig numeric comparison
    if p_nums and t_nums and len(p_nums) == len(t_nums):
        if [round_sig(x, sf) for x in p_nums] == [round_sig(y, sf) for y in t_nums]:
            correct += 1
            continue

    # ---- fallback: numeric tolerance
    if p_nums and t_nums and len(p_nums) == len(t_nums):
        if all(math.isclose(a, b, rel_tol=args.rtol, abs_tol=args.atol)
               for a, b in zip(p_nums, t_nums)):
            correct += 1
            continue

    # ---- fallback: cleaned string match
    if clean(pred_raw).lower() == clean(true_raw).lower():
        correct += 1
    else:
        mism.append((idx, pred_raw, true_raw))

# ----------------------------- report ------------------------------------
acc = correct / total if total else 0
print(f"✅ Accuracy: {correct}/{total} = {acc:.2%} "
      f"(sig-fig, rtol={args.rtol}, atol={args.atol})")

if mism:
    print("\n❌ Mismatches (first 10):")
    for idx, p, t in itertools.islice(mism, 10):
        print(f"[{idx}] Pred: {p}  |  True: {t}")
