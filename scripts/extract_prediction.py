#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(
        description="Extract full predictions and merge with dev set"
    )
    p.add_argument("solve_out", help="JSONL from call_road2all (with input+output)")
    p.add_argument("pred_json", help="Output JSON array file")
    p.add_argument("--dev",     required=True,
                   help="Original dev JSON (with question, answer, etc)")
    args = p.parse_args()

    # Load dev set (list of dicts)
    dev_items = json.load(open(args.dev, encoding="utf-8"))
    # Build map idx -> full entry
    dev_map = { int(item["index"]) : item for item in dev_items }

    # Read solve_output JSONL, extract content field
    pred_map = {}
    with open(args.solve_out, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            rec = json.loads(line)
            inp = rec.get("input", {})
            out = rec.get("output", {})
            idx = inp.get("index")
            # guard against missing or malformed
            if idx is None or out is None:
                continue
            # full assistant message content
            try:
                content = out["choices"][0]["message"]["content"]
            except Exception:
                content = ""
            pred_map[int(idx)] = content

    # Build final list
    results = []
    for idx, item in dev_map.items():
        entry = item.copy()
        entry["prediction"] = pred_map.get(idx, "")
        results.append(entry)

    # Write JSON array
    out_path = Path(args.pred_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(results)} predictions → {out_path}")

if __name__ == "__main__":
    main()
