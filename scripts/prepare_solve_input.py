#!/usr/bin/env python3
"""
prepare_solve_input.py – build solve‐stage prompts by combining question + vision details

Usage:
  python3 scripts/prepare_solve_input.py \
    --vision_out VISION_OUTPUT.jsonl \
    --dev        DEV.json \
    --output     SOLVE_INPUT.jsonl
"""
import argparse
import json
from pathlib import Path
# --- REMOVED: from string import Template ---


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dev(path):
    if path.endswith('.jsonl'):
        return load_jsonl(path)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def extract_vision_details(txt):
    """Pull out any lines beginning with “-” or “•” from the vision‐stage output."""
    bullets = []
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith('-') or line.startswith('•'):
            bullets.append(line)
    return bullets

# --- MODIFIED TEMPLATE to use .format() syntax ---
EN_TEMPLATE = r"""
{question}

Diagram details (vision):
{details}

Please solve this step-by-step:
1. Put your full reasoning inside <think>…</think>.
2. In <self-check>, verify that you have used all relevant numerical values from the 'Diagram details' section. Explicitly list the key values used and double-check your algebraic steps and final calculation.
3. Finally, in <answer>…</answer> output **either**:
   – A single number rounded to the **same** number of significant figures as in the GT  
     (no units inside the number), e.g.: `<answer>$$-6.50$$</answer>`  
   – OR a full symbolic result in LaTeX display math if the answer is an expression, e.g.:  
     `<answer>$$$$\Delta V = \frac{{(\beta_2-\beta_1)(T_2 - T_1)V_1V_2}}{{(1+\beta_1)T_2V_1 + (1+\beta_2)T_1V_2}}$$$$</answer>`
4. If the result has a physical unit, append it inside `\,\mathrm{{…}}` in the math delimiters.
5. Do **not** add any text outside these tags.

Round the numeric part to {sig_figs} significant figure(s).
""".lstrip()

# --- MODIFIED TEMPLATE to use .format() syntax ---
CN_TEMPLATE = r"""
{question}

图像要点（视觉阶段输出）:
{details}

请按以下格式作答：
1. 将全部推理放在 `<think>…</think>` 标签中。
2. 在 <self-check> 中，确认你已使用了“图像要点”中所有相关的数值。明确列出所用的关键数值，并仔细检查你的代数步骤和最终计算。
3. 最后在 `<answer>…</answer>` 中 **仅输出**：
   – 与 GT 同位数的单个数字（不含单位），如：`<answer>$$-6.50$</answer>`  
   – 或当答案是表达式时，用 LaTeX 行间公式， 如：  
     `<answer>$$$$\Delta V = \frac{{(\beta_2-\beta_1)(T_2 - T_1)V_1V_2}}{{(1+\beta_1)T_2V_1 + (1+\beta_2)T_1V_2}}$$$$</answer>`
4. 如果有物理单位，将其附加在 `\,\mathrm{{…}}` 中。
5. 不要在这些标签之外添加任何文字。

数值保留 {sig_figs} 位有效数字。
""".lstrip()


def main():
    p = argparse.ArgumentParser(
        description="Combine vision output and questions into solve‐stage prompts"
    )
    p.add_argument("--vision_out", required=True,
                   help="JSONL with vision‐stage responses")
    p.add_argument("--dev", required=True,
                   help="Original dev.json or JSONL")
    p.add_argument("--output", required=True,
                   help="Where to write solve_input.jsonl")
    args = p.parse_args()

    dev_items = load_dev(args.dev)
    vision_recs = load_jsonl(args.vision_out)

    # Build index → vision‐text map
    vision_map = {}
    for rec in vision_recs:
        inp = rec.get("input", {})
        out = rec.get("output", {})
        idx = inp.get("index")
        if idx is None or not out:
            continue
        # safe‐extract the model's content
        content = ""
        try:
            content = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        vision_map[int(idx)] = content

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for item in dev_items:
            idx      = item["index"]
            question = item["question"]
            lang     = item.get("language", "English").lower()
            sig_figs = item.get("sig_figs") or 3
            vis_txt  = vision_map.get(idx, "")
            bullets  = extract_vision_details(vis_txt)
            details  = "\n".join(bullets) if bullets else "(no vision details found)"

            tpl = EN_TEMPLATE if lang.startswith("english") else CN_TEMPLATE
            
            # --- MODIFIED from .substitute() to .format() ---
            prompt = tpl.format(
                question=question,
                details=details,
                sig_figs=sig_figs
            ).strip()

            record = {
                "index":      idx,
                "prompt":     prompt,
                "image_path": item.get("image_path", [])
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Wrote solve‐stage prompts → {out_path}")


if __name__ == "__main__":
    main()