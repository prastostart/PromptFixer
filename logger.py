import csv
import json
import os
from datetime import datetime

CSV_FILE = "prompt_logs.csv"
JSONL_FILE = "prompt_logs.jsonl"

def init_logger():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","round","candidate_id","parent_prompt","transformation","prompt","score","delta"])

def log_prompt(round_num, candidate_id, parent_prompt, transformation, prompt, score, delta=None):
    ts = datetime.utcnow().isoformat()
    row = [ts, round_num, candidate_id, parent_prompt, transformation, prompt, score, delta]
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    # JSONL
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": ts,
            "round": round_num,
            "candidate_id": candidate_id,
            "parent_prompt": parent_prompt,
            "transformation": transformation,
            "prompt": prompt,
            "score": score,
            "delta": delta
        }) + "\n")

