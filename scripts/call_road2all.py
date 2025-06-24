import os
import requests
import concurrent.futures
import time
from tqdm import tqdm
import json
import argparse
import base64

# Please contact Congkai/Zhijie to get the API key
api_key = os.getenv("ROAD2ALL_API_KEY")

if not api_key:
    raise ValueError("Missing API key. Please set the ROAD2ALL_API_KEY environment variable.")

url = "https://api.road2all.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_road2all(model_name, text, params: dict = None, timeout: int = 300, max_retries=3):
    text_dict = json.loads(text)
    prompt = text_dict["prompt"]
    if "images" in text_dict:
        content = [{ "type": "text", "text": prompt }]
        image_path_list = text_dict["images"]
        for image_path in image_path_list:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                }
            })
    else:
        content = prompt
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        **(params if params else {})
    }
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {retries + 1}): {e}. Retrying...")
            retries += 1
            time.sleep(3)
    print(f"Failed to get a valid response after {max_retries} attempts.")
    return None


def batch_call_road2all(model_name, texts, output_file, params: dict = None, timeout: int = 300, max_threads=8):
    successful_requests = 0
    failed_requests = 0
    start_time = time.time()
    total_requests = len(texts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(call_road2all, model_name, text, params, timeout): text for text in texts}

        with tqdm(total=total_requests, desc="Processing API Requests", dynamic_ncols=True) as pbar:
            for future in concurrent.futures.as_completed(futures):
                input_text = futures[future]
                result = future.result()

                combined_result = {
                    "input": input_text,
                    "output": result
                }

                if result is not None:
                    successful_requests += 1
                else:
                    failed_requests += 1

                elapsed_time = time.time() - start_time
                requests_per_second = (successful_requests + failed_requests) / elapsed_time if elapsed_time > 0 else 0
                success_rate = (successful_requests / (successful_requests + failed_requests)) * 100 if (successful_requests + failed_requests) > 0 else 0

                pbar.update(1)
                pbar.set_postfix({
                    "Success": successful_requests,
                    "Failed": failed_requests,
                    "Req/sec": f"{requests_per_second:.2f}",
                    "Success Rate": f"{success_rate:.2f}%"
                })

                if result is not None:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(combined_result, ensure_ascii=False) + '\n')


def read_prompts_from_jsonl(file_path):
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # if 'prompt' in data:
                #     prompts.append(data['prompt'])
                prompts.append(line.strip())
            except json.JSONDecodeError:
                print(f"Error decoding JSON line: {line}")
    return prompts


def get_processed_prompts(output_file):
    processed_prompts = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'input' in data:
                        processed_prompts.add(data['input'])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line in output file: {line}")
    return processed_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch API requests with retry and skip processed prompts.')
    parser.add_argument('--input_file', type=str, default="input.jsonl", help='Path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, default="results.jsonl", help='Path to the output JSONL file.')
    # parser.add_argument('--model_name', type=str, default="deepseek-r1", help='Name of the model to use.')
    parser.add_argument('--model_name', type=str, default="gpt-4.1", help='Name of the model to use.')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature parameter for the API request.')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum number of tokens for the API request.')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout for the API request in seconds.')
    parser.add_argument('--max_threads', type=int, default=12, help='Maximum number of threads for concurrent requests.')

    args = parser.parse_args()

    params = {
        "stream": False,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens
    }

    all_prompts = read_prompts_from_jsonl(args.input_file)

    processed_prompts = get_processed_prompts(args.output_file)

    unprocessed_prompts = [prompt for prompt in all_prompts if prompt not in processed_prompts]

    if unprocessed_prompts:
        batch_call_road2all(args.model_name, unprocessed_prompts, args.output_file,
                            params, timeout=args.timeout, max_threads=args.max_threads)
    else:
        print("All prompts have already been processed.")

