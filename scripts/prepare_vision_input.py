#!/usr/bin/env python3
"""
prepare_vision_input.py – build a JSONL of vision‐stage prompts

Usage:
  python3 prepare_vision_input.py IN_JSON OUT_JSONL [--sample N] [--seed S]

This script:
  - Loads your SeePhys dev‐set (JSON or JSONL)
  - (Optionally) samples N items with a fixed seed
  - Skips any item with vision_relevance="irrelevant"
  - Emits a JSONL where each line has:
      {
        "index":     <problem index>,
        "prompt":    <vision prompt (English or Chinese)>,
        "image_path": [<list of image paths>]
      }
"""

import argparse
import json
import random
from pathlib import Path

EN_VISION_TEMPLATE = (
    "Please list *only* the numeric labels and key features you see in this diagram, "
    "in bullet point form."
)

CN_VISION_TEMPLATE = (
    "请仅以要点形式列出此图中的所有数字标注和关键特征。"
)

def load_items(path: str):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.open("r", encoding="utf-8") if line.strip()]
    return json.load(p.open("r", encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser(description="Prepare vision‐stage prompts JSONL")
    parser.add_argument("input_json",  nargs="?", default="data/dev.json",
                        help="Input JSON or JSONL (default: data/dev.json)")
    parser.add_argument("output_jsonl", nargs="?", default="vision_input.jsonl",
                        help="Output JSONL (default: vision_input.jsonl)")
    parser.add_argument("--sample", type=int, default=None,
                        help="If set, randomly sample N items")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    items = load_items(args.input_json)

    # optional sampling
    if args.sample is not None:
        random.seed(args.seed)
        if args.sample > len(items):
            raise ValueError(f"--sample {args.sample} exceeds total items ({len(items)})")
        items = random.sample(items, args.sample)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for it in items:
            if it.get("vision_relevance") == "irrelevant":
                continue

            lang = str(it.get("language", "English")).lower()
            if lang.startswith("english"):
                prompt = EN_VISION_TEMPLATE
            else:
                prompt = CN_VISION_TEMPLATE

            record = {
                "index":      it["index"],
                "prompt":     prompt,
                "image_path": it.get("image_path", []),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ Wrote {count} vision‐stage prompts → {out_path}")

if __name__ == "__main__":
    main()
