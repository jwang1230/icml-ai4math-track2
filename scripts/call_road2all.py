#!/usr/bin/env python3
"""
call_road2all.py – simple API‐caller for JSONL prompts

Reads lines like:
  {"index":…, "prompt":"…", "image_path":["…"]}

and writes out:
  {"input":{…}, "output":{…API response…}}

Usage:
  python3 call_road2all.py \
    --input  vision_input.jsonl \
    --output vision_output.jsonl \
    --model  gpt-4o-mini \
    [--temperature 0.0] [--max_tokens 8192] [--threads 8]
"""

import os
import json
import time
import base64
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

API_URL = "https://api.road2all.com/v1/chat/completions"
API_KEY = os.getenv("ROAD2ALL_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing ROAD2ALL_API_KEY environment variable")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def encode_image(path: str) -> str:
    """Load image file and return a base64 data URI."""
    if not os.path.exists(path):
        alt = os.path.join("data", path)
        if os.path.exists(alt):
            path = alt
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def call_api(record: dict, model: str, params: dict, timeout: int, retries: int):
    """Call the Chat Completions API once for this record."""
    # build the content array: text + any images
    content = [{"type": "text", "text": record["prompt"]}]
    for img_path in record.get("image_path", []):
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image(img_path)}
        })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        **params
    }

    for attempt in range(1, retries+1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries:
                print(f"[error] API call failed after {retries} tries: {e}")
                return None
            time.sleep(1)

def main():
    p = argparse.ArgumentParser(description="Batch API caller for Road2All")
    p.add_argument("--input",      required=True, help="Input JSONL of prompts")
    p.add_argument("--output",     required=True, help="Output JSONL of results")
    p.add_argument("--model",      required=True, help="Model name (e.g. gpt-4o-mini or o4-mini)")
    p.add_argument("--temperature",type=float, default=0.0)
    p.add_argument("--max_tokens", type=int,   default=512)
    p.add_argument("--timeout",    type=int,   default=300)
    p.add_argument("--retries",    type=int,   default=3)
    p.add_argument("--threads",    type=int,   default=8)
    args = p.parse_args()

    params = {
        "stream": False,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens
    }

    # read all input records
    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # dispatch in parallel
    with ThreadPoolExecutor(max_workers=args.threads) as exe, \
         open(args.output, "w", encoding="utf-8") as fout, \
         tqdm(total=len(records), desc="API requests") as bar:

        futures = {
            exe.submit(call_api, rec, args.model, params, args.timeout, args.retries): rec
            for rec in records
        }

        for fut in futures:
            rec  = futures[fut]
            resp = fut.result()
            out  = {"input": rec, "output": resp}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            bar.update()

if __name__ == "__main__":
    main()
