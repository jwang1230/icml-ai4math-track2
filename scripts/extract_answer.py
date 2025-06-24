#!/usr/bin/env python3
"""
extract_answer.py
────────────────────────────────────────────────────────
Merge raw model output JSONL into a Codabench-ready prediction array.

Usage
-----
python3 extract_answer.py  RAW_JSONL  OUT_JSON  [--dev DEV_JSON]

Example
-------
python3 extract_answer.py  output_first10_o4_mini.jsonl \
                           prediction_first10_o4_mini.json \
                           --dev dev.json
"""

import argparse, json, re, sys
from pathlib import Path

# ────────────────────────── CLI ──────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("raw_jsonl", help="Raw model output JSONL")
ap.add_argument("out_json",  help="Output JSON array to write")
ap.add_argument("--dev", default="data/dev.json",
                help="Organizer dev set (default: data/dev.json)")
args = ap.parse_args()

RAW_PATH = Path(args.raw_jsonl)
OUT_PATH = Path(args.out_json)
DEV_PATH = Path(args.dev)

# ───────── helper: extract <answer> … </answer> content ─────────
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()

# ─────────────────────── load dev set ────────────────────────
try:
    dev_items = json.load(DEV_PATH.open(encoding="utf-8"))
except Exception as e:
    sys.exit(f"[error] failed to load dev set {DEV_PATH}: {e}")

dev_map = {int(d["index"]): d for d in dev_items}

# ───────────────────── parse model outputs ───────────────────
pred_map = {}
with RAW_PATH.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row   = json.loads(line)
            inp   = json.loads(row["input"])
            idx   = int(inp["index"])
            raw   = row["output"]["choices"][0]["message"]["content"]
            pred_map[idx] = extract_answer(raw)
        except Exception as e:
            print(f"[warn] line {line_no}: {e}", file=sys.stderr)

# ───────────────────── merge & sanity check ──────────────────
merged, missing, extra = [], [], []
for idx, dev_obj in dev_map.items():
    if idx in pred_map:
        obj = dict(dev_obj)
        obj["prediction"] = pred_map[idx]
        merged.append(obj)
    else:
        missing.append(idx)

extra = [i for i in pred_map if i not in dev_map]

# ─────────────────────── write output ───────────────────────
with OUT_PATH.open("w", encoding="utf-8") as fout:
    json.dump(merged, fout, ensure_ascii=False, indent=2)

print(f"✅ wrote {len(merged)} records → {OUT_PATH}")
if missing:
    print(f"⚠️  missing predictions for {len(missing)} indices "
          f"(first few: {missing[:8]})")
if extra:
    print(f"⚠️  {len(extra)} predictions had unknown indices "
          f"(first few: {extra[:8]})")
