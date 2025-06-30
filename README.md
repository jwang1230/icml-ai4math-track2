# icml-ai4math-track2

A pipeline for evaluating multimodal LLMs on physics diagram questions (ICML AI4Math Track 2).

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/icml-ai4math-track2.git
   cd icml-ai4math-track2
   ```
2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the end-to-end pipeline on the full dev set:
```bash
python3 main.py --dev data/dev.json
```
Or sample _N_ items for quick testing:
```bash
python3 main.py --dev data/dev.json --sample 10 --seed 1234
```

By default this will:
1. (Optional) sample the dev set  
2. Generate vision-stage prompts  
3. Call the vision model (e.g. `gpt-4o-mini`)  
4. Merge vision results with questions  
5. Call the solve model (e.g. `o4-mini`)  
6. Extract predictions  
7. Evaluate against ground-truth  

### Additional flags

```bash
# Model choices:
--vision_model  gpt-4o-mini
--solve_model   o4-mini

# Temperatures:
--vision_temp   0.0
--solve_temp    0.1

# Parallelism:
--threads       8
```

## Project Structure

```
.
├── data/                # dev.json, images, etc.
├── scripts/             # prepare_* and helper scripts
├── runs/                # auto-generated outputs
├── main.py              # orchestrates the full pipeline
├── requirements.txt
└── README.md
```
