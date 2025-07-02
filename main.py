#!/usr/bin/env python3
import argparse
import subprocess
import datetime
from pathlib import Path

def run(cmd: str):
    """Prints and executes a shell command."""
    print(f"→ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    p = argparse.ArgumentParser(
        description="End-to-end pipeline (vision → solve → extract → eval)"
    )
    p.add_argument("--pipeline", default="two_stage", choices=["two_stage", "single_stage"],
                   help="Choose the pipeline type to run.")
    p.add_argument("--dev",    default="data/dev.json", help="Path to full dev JSON")
    p.add_argument("--sample", type=int, default=None, help="If set, sample N items")
    p.add_argument("--seed",   type=int, default=42, help="Random seed when sampling")
    p.add_argument("--vision_model", default="o4-mini")
    p.add_argument("--solve_model",  default="o4-mini")
    p.add_argument("--vision_temp",  type=float, default=0.0)
    p.add_argument("--solve_temp",   type=float, default=0.1)
    p.add_argument("--threads",      type=int,   default=8)
    p.add_argument("--stage", default="all", choices=["all", "vision"],
                   help="For two-stage pipeline: which part to run (default: all)")
    p.add_argument("--no-eval", action="store_true",
                   help="If set, skip the final evaluation step")
    # --- NEW ARGUMENT ---
    p.add_argument("--evaluator", default="rule_based", choices=["rule_based", "llm"],
                   help="Choose the evaluation script: rule_based (fast) or llm (accurate/slow)")
    args = p.parse_args()

    # --- Setup run directory ---
    today = datetime.date.today()
    date_folder = today.strftime("%Y-%m-%d")
    sample_tag = f"S{args.sample}" if args.sample is not None else "full"
    run_name   = f"{today.strftime('%Y%m%d')}_{args.pipeline}_{args.solve_model}_{sample_tag}_seed{args.seed}"
    outdir     = Path("runs") / date_folder / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"✅ All outputs for this run will be saved in: {outdir}")

    # --- (Optional) sample dev set ---
    dev_input = Path(args.dev)
    if args.sample is not None:
        sampled_path = outdir / "dev_sample.json"
        run(f"python3 scripts/sample_dev.py {args.dev} {sampled_path} --sample {args.sample} --seed {args.seed}")
        dev_input = sampled_path
        
    # --- Conditional Pipeline Logic ---
    if args.pipeline == "two_stage":
        print("\n--- Running Two-Stage Pipeline ---")
        # STAGE 1: VISION
        vision_in  = outdir / "vision_input.jsonl"
        vision_out = outdir / "vision_output.jsonl"
        
        run(f"python3 scripts/prepare_vision_input.py {dev_input} {vision_in}")
        run(
            f"python3 scripts/call_road2all.py "
            f"--input {vision_in} --output {vision_out} "
            f"--model {args.vision_model} "
            f"--temperature {args.vision_temp} --max_tokens 8192 --threads {args.threads}"
        )

        if args.stage == "vision":
            print(f"\n✅ VISION-ONLY pipeline finished — outputs in {outdir}")
            return

        # STAGE 2: SOLVE
        print("\n--- Continuing to Solve Stage ---")
        solve_in = outdir / "solve_input.jsonl"
        solve_out = outdir / "solve_output.jsonl"
        
        run(f"python3 scripts/prepare_solve_input.py --vision_out {vision_out} --dev {dev_input} --output {solve_in}")
        run(
            f"python3 scripts/call_road2all.py "
            f"--input {solve_in} --output {solve_out} "
            f"--model {args.solve_model} "
            f"--temperature {args.solve_temp} --max_tokens 16384 --threads {args.threads}"
        )
        
    else: # single_stage
        print("\n--- Running Single-Stage Pipeline ---")
        solve_in = outdir / "solve_input.jsonl"
        solve_out = outdir / "solve_output.jsonl"
        
        run(f"python3 scripts/prepare_single_stage_input.py {dev_input} {solve_in}")
        run(
            f"python3 scripts/call_road2all.py "
            f"--input {solve_in} --output {solve_out} "
            f"--model {args.solve_model} "
            f"--temperature {args.solve_temp} --max_tokens 16384 --threads {args.threads}"
        )

    # --- Common final steps for both pipelines ---
    pred = outdir / "prediction.json"
    run(f"python3 scripts/extract_prediction.py {solve_out} {pred} --dev {dev_input}")

    if not args.no_eval:
        # --- MODIFIED: Conditional Evaluation Logic ---
        if args.evaluator == 'llm':
            print("\n--- Evaluating with LLM Judge ---")
            # Pass threads for faster parallel evaluation and define the output log file
            eval_log = outdir / "evaluation_details.json"
            run(f"python3 scripts/evaluate_llm.py {pred} {dev_input} --threads {args.threads} --output_file {eval_log}")
        else: # rule_based
            print("\n--- Evaluating with Rule-Based Scorer ---")
            run(f"python3 scripts/evaluate.py {pred} {dev_input}")
        
        print(f"\n✅ Pipeline finished with evaluation — all outputs in {outdir}")
    else:
        print(f"\n✅ Pipeline finished without evaluation — all outputs in {outdir}")


if __name__ == "__main__":
    main()