# ICML AI4Math Track-2 Pipeline

This repository contains a complete end-to-end pipeline for the SeePhys AI4Math Track-2 challenge, including data preparation, prompt construction, inference, answer extraction, and evaluation.

## Project Structure

```
icml-ai4math-track2/
├─ data/                 # Original dev/test JSON and images
│  ├─ dev.json
│  └─ images/
│      └─ *.png
│
├─ scripts/              # Python scripts for each stage
│  ├─ prepare_input.py   # Build JSONL prompts
│  ├─ call_road2all.py   # Invoke the API for inference
│  ├─ extract_answer.py  # Parse model outputs into prediction.json
│  ├─ evaluate.py        # Dev-set evaluator with sig-fig logic
│  └─ build_submission.py# Merge predictions into submission format
│
├─ runs/                 # Experiment outputs (organized by date/model)
│  └─ YYYY-MM-DD_model/
│      ├─ input.jsonl
│      ├─ output.jsonl
│      ├─ prediction.json
│      └─ metrics.txt
│
├─ notebooks/            # Exploratory analysis (optional)
├─ .env                  # Local API keys (ignored)
├─ .gitignore            # Ignore rules
├─ requirements.txt      # Python dependencies
└─ README.md             # This file
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys.

## Usage

### 1. Prepare Prompts

```bash
python3 scripts/prepare_input.py data/dev.json runs/<tag>/input.jsonl
```

### 2. Run Inference

```bash
python3 scripts/call_road2all.py \
    --input_file runs/<tag>/input.jsonl \
    --output_file runs/<tag>/output.jsonl \
    --model_name o4-mini --temperature 0.2
```

### 3. Extract Answers

```bash
python3 scripts/extract_answer.py \
    runs/<tag>/output.jsonl \
    runs/<tag>/prediction.json \
    --dev data/dev.json
```

### 4. Evaluate Predictions

```bash
python3 scripts/evaluate.py \
    runs/<tag>/prediction.json data/dev.json \
    --rtol 0.01 --atol 0.001 | tee runs/<tag>/metrics.txt
```

### 5. Build Submission

```bash
python3 scripts/build_submission.py \
    data/dev.json runs/<tag>/prediction.json submission.json
```

## Experiment Management

* Store each run under `runs/<date_model>/`.
* Use `gitignore` to skip large outputs and secrets.

## Contributing

Feel free to open issues or pull requests for improvements.

---

Good luck with the AI4Math Track-2 challenge!

