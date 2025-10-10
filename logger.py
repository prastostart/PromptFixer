# logger.py
import csv
import json
import os
from datetime import datetime

CSV_FILE = "prompt_logs.csv"
BEST_FILE = "best_prompts.csv"
JSONL_FILE = "prompt_logs.jsonl"

def init_logger():
    # Main log containing all evaluated prompts and outputs
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Fields: timestamp, round, candidate_id, prompt, output, root_cause_fixed, acr_score
            writer.writerow(["timestamp", "round", "candidate_id", "prompt", "output", "root_cause_fixed", "acr_score"])

    # File to store best prompts for reuse
    if not os.path.exists(BEST_FILE):
        with open(BEST_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "best_prompt", "acr_score"])

def log_prompt(round_num, candidate_id, prompt, output, root_cause_fixed, acr_score):
    ts = datetime.utcnow().isoformat()
    row = [ts, round_num, candidate_id, prompt, output, root_cause_fixed, acr_score]
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    # JSONL backup
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": ts,
            "round": round_num,
            "candidate_id": candidate_id,
            "prompt": prompt,
            "output": output,
            "root_cause_fixed": root_cause_fixed,
            "acr_score": acr_score
        }) + "\n")

def log_best_prompt(prompt, acr_score):
    ts = datetime.utcnow().isoformat()
    with open(BEST_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, prompt, acr_score])

