#!/usr/bin/env python3
"""
prepare_vision_input.py – build a JSONL of vision‐stage prompts

Usage:
  python3 scripts/prepare_vision_input.py IN_JSON OUT_JSONL [--sample N] [--seed S]

This script now uses category-specific prompts to improve vision extraction.
"""

import argparse
import json
import random
from pathlib import Path

# --- NEW: English Prompt Templates by Category ---
EN_PROMPT_TEMPLATES = {
    "circuit_diagram":
        "This is an electrical circuit diagram. Please list all components (e.g., resistors, capacitors, op-amps, batteries), "
        "their labels (e.g., R1, C2), their values (e.g., 10Ω, 5V, 1.2kΩ), and any marked currents or nodes.",
    "capacitance_resistance":
        "This is a diagram of a capacitor or resistor network. Please list all components, their labels, their values, "
        "and key geometric features like radii (R, a), length (l), and separation (d).",
    "static_force_analysis":
        "This is a static equilibrium or force analysis diagram. Please list all objects, force vectors (e.g., F, T, N, mg), "
        "masses, angles, lengths, pivot points, and coordinate axes in bullet point form.",
    "spring_force":
        "This is a diagram involving springs. Please list all masses, springs, spring constants (k), displacements (x), "
        "and equilibrium positions in bullet point form.",
    "circular_motion":
        "This is a diagram of circular motion. Please list all masses, radii (r, R), angles (θ), angular velocities (ω), "
        "and forces involved (e.g., tension, gravity).",
    "projectile_motion":
         "This is a projectile motion diagram. Please list initial velocity (v₀), launch angle (θ₀), heights (h), and horizontal distances (a, b).",
    "simple_harmonic_motion":
        "This is a diagram of simple harmonic motion. Please list amplitudes, periods, frequencies, masses, and spring constants.",
    "optical_path":
        "This is an optics diagram. Please list all optical components (lenses, mirrors, prisms, slits), light rays, "
        "angles (θ, α), wavelengths (λ), refractive indices (n), and labeled distances (f, d, l).",
    "coordinate_system":
        "This is a graph. Please identify the labels and units for the x-axis and y-axis. List key coordinates, intercepts, "
        "and describe the shape or trend of the plotted line(s).",
    "electromagnetic_field":
        "This is a diagram of electric or magnetic fields. Please list all charges (e.g., +Q, -e), current densities (j), "
        "field lines (B, E), and key distances or radii.",
    "charge_distribution":
        "This is a diagram of charge distribution. Please list all charges (Q, q), charge densities (σ, ρ), radii (R, r, a), "
        "and labeled points or distances.",
    "thermodynamics":
        "This is a thermodynamics diagram. Please list all labeled states, temperatures (T), pressures (P), volumes (V), "
        "and quantities of substance (e.g., n, m).",
    # Default prompt for any other categories
    "default":
        "Please list *only* the numeric labels and key features you see in this diagram, in bullet point form."
}

# --- NEW: Chinese Prompt Templates by Category ---
CN_PROMPT_TEMPLATES = {
    "circuit_diagram":
        "这是一张电路图。请以要点形式列出所有元件（例如：电阻、电容、运放、电池），它们的标签（例如：R1, C2），"
        "它们的数值（例如：10Ω, 5V, 1.2kΩ），以及任何标记的电流或节点。请用中文回答。",
    "capacitance_resistance":
        "这是一张电容或电阻网络图。请以要点形式列出所有元件、标签、数值以及关键几何特征，如半径（R, a）、"
        "长度（l）和间距（d）。请用中文回答。",
    "static_force_analysis":
        "这是一张静力平衡或力学分析图。请以要点形式列出所有物体、力矢量（例如：F, T, N, mg）、质量、角度、"
        "长度、支点和坐标轴。请用中文回答。",
    "spring_force":
        "这是一张涉及弹簧的图。请以要点形式列出所有质量、弹簧、劲度系数（k）、位移（x）和平衡位置。请用中文回答。",
    "circular_motion":
        "这是一张圆周运动图。请以要点形式列出所有质量、半径（r, R）、角度（θ）、角速度（ω）和涉及的力"
        "（例如：张力，重力）。请用中文回答。",
    "projectile_motion":
        "这是一张抛体运动图。请以要点形式列出初速度（v₀）、抛射角（θ₀）、高度（h）和水平距离（a,b）。请用中文回答。",
    "optical_path":
        "这是一张光学路径图。请以要点形式列出所有光学元件（透镜、反射镜、棱镜、狭缝）、光线、角度（θ, α）、"
        "波长（λ）、折射率（n）和标记的距离（f, d, l）。请用中文回答。",
    "coordinate_system":
        "这是一张图表。请识别x轴和y轴的标签和单位。列出关键坐标点、截距，并描述曲线的形状或趋势。请用中文回答。",
    "electromagnetic_field":
        "这是一张电场或磁场图。请以要点形式列出所有电荷（例如：+Q, -e）、电流密度（j）、场线（B, E）和关键距离或半径。请用中文回答。",
    "charge_distribution":
        "这是一张电荷分布图。请以要点形式列出所有电荷（Q, q）、电荷密度（σ, ρ）、半径（R, r, a）和标记的点或距离。请用中文回答。",
    "thermodynamics":
        "这是一张热力学图。请以要点形式列出所有标记的状态、温度（T）、压力（P）、体积（V）和物质的量（例如：n, m）。请用中文回答。",
    # Default prompt for any other categories
    "default":
        "请仅以要点形式列出此图中的所有数字标注和关键特征。请用中文回答。"
}


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

            # --- MODIFIED LOGIC ---
            # Get the category, fall back to "default" if not present
            category = it.get("img_category", "default")
            lang = str(it.get("language", "English")).lower()

            if lang.startswith("english"):
                # Select prompt from the dictionary, falling back to the default one
                prompt = EN_PROMPT_TEMPLATES.get(category, EN_PROMPT_TEMPLATES["default"])
            else:
                # Select prompt from the Chinese dictionary
                prompt = CN_PROMPT_TEMPLATES.get(category, CN_PROMPT_TEMPLATES["default"])

            record = {
                "index":      it["index"],
                "prompt":     prompt,
                "image_path": it.get("image_path", []),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ Wrote {count} category-specific vision-stage prompts → {out_path}")


if __name__ == "__main__":
    main()