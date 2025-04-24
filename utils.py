import argparse
import sys
import config
import requests
import json
from datasets import load_dataset
from datetime import datetime
import os

def parse_method_from_argv() -> str:
    if len(sys.argv) != 2:
        print("Provide python <script>.py <method>.py")
        sys.exit(1)

    script_name = sys.argv[1]
    method = os.path.splitext(os.path.basename(script_name))[0]
    return method

def log_failure(description: str):
    print(description)
    with open(f"{config.FAILURES}.log", "a") as log:
        log.write(f"{description}\n")

def get_completion(system_prompt: str, user_prompt: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "api-key": config.KEY
    }
    payload = {
        "model": config.MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": config.TEMP
    }
    response = requests.post(
        config.URL,
        headers=headers,
        data=json.dumps(payload)
    )
    response.raise_for_status()
    return response.json()

def get_arguments() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_index", type=int, help="Provide 0-based trial index.")
    parser.add_argument("problem_index", type=int, help="Provide 0-based HumanEval problem index.")
    args = parser.parse_args()

    if args.trial_index < 0 or args.trial_index >= config.NUM_TRIALS:
        print(f"Trial index must be between 0 and {config.NUM_TRIALS - 1} inclusive.")
        sys.exit(1)

    if args.problem_index < 0 or args.problem_index >= config.NUM_PROBLEMS:
        print(f"Problem index must be between 0 and {config.NUM_PROBLEMS - 1} inclusive.")
        sys.exit(1)

    return args.trial_index, args.problem_index

def load_problem(problem_index: int) -> dict:
    ds = load_dataset(config.DATASET, split="test")
    return ds[problem_index]

def create_directory(method: str, trial_number: int) -> str:
    directory = f"{config.RESULTS}/{method}/{trial_number}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_name() -> str:
    return os.path.splitext(os.path.basename(sys.argv[0]))[0]

def get_prompt(method: str) -> str:
    prompt_path = os.path.join(config.PROMPTS, f"{method}.txt")
    with open(prompt_path, "r") as f:
        return f.read()

def build_result(data, system_prompt, user_prompt, problem_id, problem_data, path, method, trial, start, end, usage) -> dict:
    dt1 = datetime.fromisoformat(start)
    dt2 = datetime.fromisoformat(end)
    delta_seconds = (dt2 - dt1).total_seconds()

    return {
        "completion": data["choices"][0]["message"]["content"],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "problem_id": problem_id,
        "problem_data": problem_data,
        "path": path,
        "method": method,
        "trial": trial,
        "time_taken": delta_seconds,
        "time_start": start,
        "time_end": end,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "raw": data,
    }
