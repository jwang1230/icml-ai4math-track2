#!/usr/bin/env python3
import argparse
import subprocess
import datetime
from pathlib import Path

def run(cmd: str):
    print(f"→ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    p = argparse.ArgumentParser(
        description="End-to-end pipeline (vision → solve → extract → eval)"
    )
    p.add_argument("--dev",    default="data/dev.json", help="Path to full dev JSON")
    p.add_argument("--sample", type=int, default=None, help="If set, sample N items")
    p.add_argument("--seed",   type=int, default=42, help="Random seed when sampling")
    p.add_argument("--vision_model", default="gpt-4o")
    p.add_argument("--solve_model",  default="o4-mini")
    p.add_argument("--vision_temp",  type=float, default=0.0)
    p.add_argument("--solve_temp",   type=float, default=0.1)
    p.add_argument("--threads",      type=int,   default=8)
    args = p.parse_args()

    today = datetime.date.today()
    date_folder = today.strftime("%Y-%m-%d")
    sample_tag = f"S{args.sample}" if args.sample is not None else "full"
    run_name   = f"{today.strftime('%Y%m%d')}_{args.solve_model}_{sample_tag}_seed{args.seed}"
    outdir     = Path("runs") / date_folder / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) (optional) sample dev
    dev_input = Path(args.dev)
    if args.sample is not None:
        sampled = outdir / "dev_sample.json"
        run(f"python3 scripts/sample_dev.py {args.dev} {sampled} --sample {args.sample} --seed {args.seed}")
        dev_input = sampled

    # 2) vision prompts
    vision_in  = outdir / "vision_input.jsonl"
    run(f"python3 scripts/prepare_vision_input.py {dev_input} {vision_in}")

    # 3) vision call — use --input_file / --output_file
    vision_out = outdir / "vision_output.jsonl"
    run(
        f"python3 scripts/call_road2all.py "
        f"--input {vision_in} --output {vision_out} "
        f"--model {args.vision_model} "
        f"--temperature {args.vision_temp} --max_tokens 512 --threads {args.threads}"
    )

    # 4) solve prompts
    solve_in = outdir / "solve_input.jsonl"
    run(
        f"python3 scripts/prepare_solve_input.py "
        f"--vision_out {vision_out} --dev {dev_input} --output {solve_in}"
    )

    # 5) solver call — also fix flags here
    solve_out = outdir / "solve_output.jsonl"
    run(
        f"python3 scripts/call_road2all.py "
        f"--input {solve_in} --output {solve_out} "
        f"--model {args.solve_model} "
        f"--temperature {args.solve_temp} --max_tokens 8192 --threads {args.threads}"
    )

    # 6) extract
    pred = outdir / "prediction.json"
    run(f"python3 scripts/extract_prediction.py {solve_out} {pred} --dev {dev_input}")

    # 7) evaluate
    run(f"python3 scripts/evaluate.py {pred} {dev_input}")

    print(f"\n✅ Pipeline finished — all outputs in {outdir}")

if __name__ == "__main__":
    main()
