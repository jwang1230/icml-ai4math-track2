#!/usr/bin/env python3
"""
prepare_single_stage_input.py – build end-to-end prompts for a single API call.
This version uses a hyper-contextual and localized prompt wrapper to guide the model.
"""
import argparse
import json
import random
from pathlib import Path

def load_items(path: str):
    """Loads items from a JSON or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

# --- English Mappings ---
LEVEL_MAP_EN = {
    1: "Middle School",
    2: "High School",
    3: "Physics Olympiad (Beginner)",
    4: "Physics Olympiad (Advanced)",
    5: "Undergraduate (Calculus-based)",
    6: "Senior Undergraduate",
    7: "Master's Level",
    8: "PhD Qualifying Exam Level"
}

# --- Chinese Mappings for Localization ---
LEVEL_MAP_CN = {
    1: "初中水平",
    2: "高中水平",
    3: "物理竞赛（初级）",
    4: "物理竞赛（高级）",
    5: "大学本科（基于微积分）",
    6: "高年级本科",
    7: "硕士研究生水平",
    8: "博士资格考试水平"
}

SUBJECT_MAP_CN = {
    "CM": "经典力学",
    "EM": "电磁学",
    "TSM": "热力学与统计力学",
    "OPT": "光学",
    "AMONP": "原子、分子与核物理",
    "ACG": "天文学、宇宙学与引力"
}

IMG_CATEGORY_MAP_CN = {
    'astrophysics': '天体物理图',
    'atomic_physics': '原子物理图',
    'capacitance_resistance': '电容电阻图',
    'charge_distribution': '电荷分布图',
    'circuit_diagram': '电路图',
    'circular_motion': '圆周运动图',
    'coordinate_system': '坐标系图/图表',
    'electromagnetic_field': '电磁场图',
    'linear_motion': '直线运动图',
    'optical_path': '光路图',
    'photoelectric_effect': '光电效应图',
    'projectile_motion': '抛体运动图',
    'quantum_mechanics': '量子力学图',
    'relativity_gravity': '相对论与引力图',
    'simple_harmonic_motion': '简谐振动图',
    'spring_force': '弹簧力学图',
    'static_force_analysis': '静力学分析图',
    'thermodynamics': '热力学图',
    'wave_motion': '波动图'
}


# Hyper-Contextual English Template using .format() syntax
EN_TEMPLATE = r"""
**Your Role:** You are an expert physicist and data analyst. Your task is to solve a physics problem that is described by the text and an accompanying diagram.

**Problem Context:**
- Subject: {subject}
- Difficulty Level: {level_desc}
- Diagram Type: {img_category}

**Process to Follow:**
1.  **Analyze the Diagram First:** Given the diagram is a '{img_category}', pay special attention to its specific conventions and the components relevant to a '{subject}' problem. Explicitly list all key features, labels, and numerical values.
2.  **Synthesize and Solve:** Based on the problem's context (Level: {level_desc}), apply the appropriate physical principles and mathematical methods to the information you have gathered.

**Problem Statement:**
{question}

**Required Output Format:**
You must structure your entire output using the following XML tags:
1.  **<think>...</think>:** Place your complete reasoning here, including your vision analysis and step-by-step solution.
2.  **<self-check>...</self-check>:** In this tag, verify your key calculations and the consistency of your answer.
3.  **<answer>...</answer>:** In this tag, provide ONLY the final answer in the required format.

Please adhere strictly to this format. Round the final numeric part to {sig_figs} significant figure(s).
""".lstrip()

# Hyper-Contextual Chinese Template using .format() syntax
CN_TEMPLATE = r"""
**你的角色:** 你是一位资深的物理学家和数据分析师。你的任务是解决一个由文本和图表共同描述的物理问题。

**问题背景:**
- 学科: {subject}
- 难度等级: {level_desc}
- 图表类型: {img_category}

**必须遵循的流程:**
1.  **首先分析图表:** 鉴于图表是“{img_category}”，请特别注意其特有的惯例以及与“{subject}”问题相关的元件。明确列出所有关键特征、标签和数值。
2.  **综合与求解:** 根据问题背景（等级: {level_desc}），将适当的物理原理和数学方法应用于你收集到的信息。

**问题描述:**
{question}

**要求的输出格式:**
你必须使用以下XML标签来构建你的全部输出：
1.  **<think>...</think>:** 将你完整的推理过程（包括视觉分析和解题步骤）放在这里。
2.  **<self-check>...</self-check>:** 在此标签中，核对你的关键计算和答案的一致性。
3.  **<answer>...</answer>:** 在此标签中，仅提供所需格式的最终答案。

请严格遵守此格式。将最终数值部分四舍五入到 {sig_figs} 位有效数字。
""".lstrip()


def main():
    p = argparse.ArgumentParser(
        description="Prepare single-stage prompts (question + image)."
    )
    p.add_argument("input_json",  nargs="?", default="data/dev.json",
                   help="Input JSON or JSONL (default: data/dev.json)")
    p.add_argument("output_jsonl", nargs="?", default="solve_input.jsonl",
                   help="Output JSONL (default: solve_input.jsonl)")
    p.add_argument("--sample", type=int, default=None,
                   help="If set, randomly sample N items")
    p.add_argument("--seed",   type=int, default=42,
                   help="Random seed (default: 42)")
    args = p.parse_args()

    items = load_items(args.input_json)

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
            
            sig_figs = it.get("sig_figs") or 3
            subject_key = it.get("subject", "General Physics")
            img_category_key = it.get("img_category", "diagram")
            level_int = it.get("level")

            if lang.startswith("english"):
                tpl = EN_TEMPLATE
                level_desc = LEVEL_MAP_EN.get(level_int, "Intermediate")
                subject_desc = subject_key
                img_category_desc = img_category_key
            else:
                tpl = CN_TEMPLATE
                level_desc = LEVEL_MAP_CN.get(level_int, "中级")
                subject_desc = SUBJECT_MAP_CN.get(subject_key, subject_key)
                img_category_desc = IMG_CATEGORY_MAP_CN.get(img_category_key, img_category_key)

            prompt = tpl.format(
                question=it["question"], 
                sig_figs=sig_figs,
                subject=subject_desc,
                level_desc=level_desc,
                img_category=img_category_desc
            )
            
            record = {
                "index":      it["index"],
                "prompt":     prompt.strip(),
                "image_path": it.get("image_path", [])
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ Wrote {count} hyper-contextual and localized single-stage prompts → {out_path}")


if __name__ == "__main__":
    main()