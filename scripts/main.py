import os
import json
import base64
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

client = OpenAI(api_key=api_key)

# Helper: encode image to base64
def encode_image(image_path):
    abs_path = os.path.join("data", image_path)  # prepend "data/"
    with open(abs_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Helper: extract content inside <answer> tags
def extract_answer_tag(response):
    import re
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

# Load dev set
with open("data/dev.json", "r", encoding="utf-8") as f:
    dev_data = json.load(f)

output_path = "predictions-dev-test.json"
results = []

for item in tqdm(dev_data):
    question = item["question"]
    caption = item["caption"]
    img_paths = item["image_path"]
    img_category = item["img_category"]
    language = item["language"]
    subject = item["subject"]
    level = item["level"]
    sig_figs = item["sig_figs"]

    if language != "English":
        continue
    if item["vision_relevance"] == "irrelevant":
        continue

    system_msg = (
        f"You are a {subject} expert tutoring a grade-{level} student. "
        f"Solve the problem step by step and give the final numeric answer "
        f"rounded to {sig_figs} significant figure(s)."
    )

    prompt = (
        f"{question}\n"
        "Please answer this question with reasoning. "
        "First output your reasoning process in <think></think> tags, "
        # "then give the final result in <answer></answer>."
        "then give the final result in \\boxed{}"
    )

    # Build message
    user_parts = [{"type": "text", "text": prompt}]
    user_parts.append({"type": "text", "text": f"Diagram type: {img_category}"})
    user_parts.append({"type": "text", "text": f"Caption: {caption}"})

    for img_path in img_paths:
        b64 = encode_image(img_path)
        user_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:images/png;base64,{b64}"
            }
        })

    # Retry inference
    max_retries = 5
    retry_delay = 30
    attempt = 0
    response_text = ""

    while attempt < max_retries:
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_parts}
                ],
                temperature=0,
            )
            response_text = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            attempt += 1
            print(f"[Retry {attempt}] {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                response_text = "ERROR: Max retries reached."

    # Extract prediction from <answer> tag
    extracted_answer = extract_answer_tag(response_text)

    results.append({
        "index": item["index"],
        "question": question,
        "ground_truth": item["answer"],
        "prediction_raw": response_text,
        "prediction_extracted": extracted_answer,
    })
    break

# Write output
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Predictions saved to {output_path}")
# End of script