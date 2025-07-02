#!/usr/bin/env python3
"""
evaluate_llm.py – Use an LLM as a judge to score model predictions.
Saves detailed mismatch information to a file and prints a concise summary.
"""
import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tqdm import tqdm

# The Judge Prompt Template remains the same
JUDGE_PROMPT_TEMPLATE = r"""
**Your Role:** You are an expert Physics Professor and a strict but fair grader. Your task is to evaluate a model's predicted answer against a ground truth solution.

**Instructions:**
You must determine if the predicted answer is correct. A prediction is considered correct if it is mathematically or physically equivalent to the ground truth, even if the formatting is different. Follow these rules:
1.  **Mathematical Equivalence:** Equivalent formulas (e.g., sin(x) vs. -cos(x + pi/2)) are correct.
2.  **Numerical Tolerance:** Allow for minor rounding differences (e.g., 209.9 is equivalent to 210).
3.  **Unit Equivalence:** Answers with different but equivalent units are correct (e.g., 250 nm is equivalent to 0.25 micrometers).
4.  **Formatting:** Ignore minor LaTeX formatting differences (e.g., \mathrm vs. plain text, or extra approximations like ≈1.21R if the main symbolic part is correct).
5.  **Partial Credit is Failure:** The answer must be fully correct to be marked as correct. A symbolic formula without the required numerical calculation is incorrect.

**The Problem's Solution:**
- **Ground Truth:** {ground_truth}
- **Model's Prediction:** {prediction}

**Your Task:**
Is the "Model's Prediction" correct based on the rules above? Respond ONLY with a valid JSON object in the following format and nothing else:
{{
  "is_correct": boolean,
  "reasoning": "A brief, one-sentence explanation for your decision."
}}
"""

# Client initialization remains the same
client = OpenAI(
    api_key=os.environ.get("ROAD2ALL_API_KEY"),
    base_url="https://api.road2all.com/v1"
)

def get_llm_judgment(gt, pred, model_name="o4-mini"):
    """Calls the LLM to get a judgment."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(ground_truth=str(gt), prediction=str(pred))
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        judgment_str = response.choices[0].message.content
        return json.loads(judgment_str)
    except Exception as e:
        return {"is_correct": False, "reasoning": f"API or JSON parsing error: {e}"}

def main():
    p = argparse.ArgumentParser(description="Evaluate predictions using an LLM judge.")
    p.add_argument("pred_path", help="Path to prediction.json file")
    p.add_argument("gt_path", help="Path to the dev_sample.json or dev.json with ground-truth answers")
    p.add_argument("--threads", type=int, default=16, help="Number of parallel API calls")
    p.add_argument("--output_file", help="Path to save detailed evaluation results. Defaults next to pred_path.")
    args = p.parse_args()

    pred_path = Path(args.pred_path)
    gt_path = Path(args.gt_path)

    output_path = Path(args.output_file) if args.output_file else pred_path.with_name("evaluation_details.json")

    # Load predictions and ground truths
    with pred_path.open("r", encoding="utf-8") as f:
        preds = json.load(f)
    with gt_path.open("r", encoding="utf-8") as f:
        gts = json.load(f)
        
    gt_map = {item["index"]: item["answer"] for item in gts}
    
    tasks = [pred_item for pred_item in preds if pred_item.get("index") in gt_map]

    correct_count = 0
    evaluation_results = []
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor, \
         tqdm(total=len(tasks), desc="LLM Evaluating") as bar:
         
        future_to_task = {
            executor.submit(get_llm_judgment, gt_map[task["index"]], task.get("prediction", "")): task 
            for task in tasks
        }
        
        for future in future_to_task:
            task = future_to_task[future]
            judgment = future.result()
            
            is_correct = judgment.get("is_correct", False) if judgment else False
            if is_correct:
                correct_count += 1
            
            evaluation_results.append({
                "index": task["index"],
                "is_correct": is_correct,
                "prediction": task.get("prediction", ""),
                "ground_truth": gt_map[task["index"]],
                "judgement_reasoning": judgment.get("reasoning", "No reasoning provided.")
            })
            bar.update()

    total_count = len(tasks)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Save detailed results to the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(evaluation_results, key=lambda x: x['index']), f, ensure_ascii=False, indent=2)

    # --- MODIFIED: Always print the concise summary ---
    print(f"\n✅ Detailed evaluation results saved to: {output_path}")
    print(f"✅ LLM-Judged Accuracy: {correct_count}/{total_count} = {accuracy:.2%}")

if __name__ == "__main__":
    main()