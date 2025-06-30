#!/usr/bin/env python3
"""
sample_dev.py – sample N items from the SeePhys dev‐set

Usage:
  python3 sample_dev.py IN_JSON OUT_JSON --sample N [--seed S]

This script:
  - Loads your full dev set (JSON array or JSONL)
  - Randomly samples N entries (with a fixed seed, default 42)
  - Writes them as a JSON **array** to OUT_JSON
"""

import argparse, json, random
from pathlib import Path

def load_items(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.open("r", encoding="utf-8") if l.strip()]
    return json.load(p.open("r", encoding="utf-8"))

def main():
    p = argparse.ArgumentParser(description="Sample N items from dev set")
    p.add_argument("input",  help="Full dev JSON or JSONL")
    p.add_argument("output", help="Where to write sampled dev (JSON array)")
    p.add_argument("--sample", type=int, required=True,
                   help="Number of items to sample")
    p.add_argument("--seed",   type=int, default=42,
                   help="Random seed (default 42)")
    args = p.parse_args()

    items = load_items(args.input)
    if args.sample > len(items):
        raise ValueError(f"--sample {args.sample} > {len(items)} available")
    random.seed(args.seed)
    sampled = random.sample(items, args.sample)

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(sampled)} sampled items → {outp}")

if __name__=="__main__":
    main()
