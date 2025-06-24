#!/usr/bin/env python3
"""
prepare_input.py – build a Road2All-ready prompt file for SeePhys / AI4Math

Run
----
python3 prepare_input.py  IN_JSON  OUT_JSONL
# defaults: IN_JSON=data/dev.json  OUT_JSONL=input.jsonl
"""

import argparse, json, textwrap, re

# ────────────────────────────── CLI ────────────────────────────────────
ap = argparse.ArgumentParser(description="Create prompt JSONL for Road2All")
ap.add_argument("input_json",  nargs="?", default="data/dev.json",
                help="Input JSON/JSONL (default: data/dev.json)")
ap.add_argument("output_jsonl", nargs="?", default="input.jsonl",
                help="Output JSONL file (default: input.jsonl)")
args = ap.parse_args()

# ──────────────────────── helpers & loader ─────────────────────────────
def load(path: str):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return json.load(open(path, encoding="utf-8"))

dev_items = load(args.input_json)

# ─────────────────────── prompt templates ──────────────────────────────
EN_TEMPLATE = """\
{question}

Here is a description of the diagram:
{caption}

Please solve the problem step-by-step.  Put your reasoning inside <think></think> tags.

Then, in <answer></answer>, write **only the final result in LaTeX math**:

  • If the answer is a pure number, output just the number  
    (e.g. `<answer>$-6.5$</answer>`).

  • If the answer has a physical unit (V, N, T, J, kg·m/s², etc.), output  
    `<value>\\,\\mathrm{{unit}}` inside the math delimiters  
    (e.g. `<answer>$-2.6\\,\\mathrm{{V}}$</answer>`).

Round the numeric part to {sig_figs} significant figure(s).  
Do **not** add explanatory text outside the tags.
"""

CN_TEMPLATE = """\
{question}

这是图像的描述：
{caption}

请推理并回答。推理写在<think></think>标签内。

然后在<answer></answer>标签内仅写**最终结果（LaTeX 数学格式）**：

  • 若答案是纯数字，只写数字，如 `<answer>$-6.5$</answer>`  
  • 若答案包含物理单位（V、N、T、J 等），写成  
    `<answer>$<数值>\\,\\mathrm{{单位}}$</answer>`，例如 `$-2.6\\,\\mathrm{{V}}$`

数值部分需保留 {sig_figs} 位有效数字，不要额外说明文字。
"""

newline_clean = re.compile(r"\s+")

# ───────────────────────── build + write ───────────────────────────────
written = 0
with open(args.output_jsonl, "w", encoding="utf-8") as fout:
    for item in dev_items:
        if item.get("vision_relevance") == "irrelevant":
            continue  # skip non-visual problems

        idx       = item["index"]
        lang      = item.get("language", "English")
        sig_figs  = str(item.get("sig_figs") or 3)
        caption   = newline_clean.sub(" ", item.get("caption", ""))
        question  = item["question"]
        images    = item["image_path"]

        if lang == "English":
            prompt = textwrap.dedent(EN_TEMPLATE).format(
                question=question, caption=caption, sig_figs=sig_figs
            ).strip()
        else:
            prompt = textwrap.dedent(CN_TEMPLATE).format(
                question=question, caption=caption, sig_figs=sig_figs
            ).strip()

        fout.write(json.dumps({
            "index": idx,
            "prompt": prompt,
            "image_path": images
        }, ensure_ascii=False) + "\n")
        written += 1

print(f"✅ Wrote {written} prompts → {args.output_jsonl} (source: {args.input_json})")
