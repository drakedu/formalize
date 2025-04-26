import os
import json
import random
import pandas as pd
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness
import config
import utils

# Choose a completion by self-consistency voting among first j completions.
def vote_among_first_j(completions, j):
    subset = completions[:j]
    counter = Counter(subset)
    most_common = counter.most_common()
    max_votes = most_common[0][1]
    top_candidates = [comp for comp, count in most_common if count == max_votes]
    return random.choice(top_candidates)

# Evaluate a single completion using HumanEval with retries and detailed error logging.
def evaluate_completion(method, trial_index, problem_index, j, completion, problem_data):
    task_id = problem_data["task_id"]

    for attempt in range(config.NUM_RETRIES):
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp_samples:
                tmp_samples.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")
                tmp_samples_path = tmp_samples.name

            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp_problem:
                tmp_problem.write(json.dumps(problem_data) + "\n")
                tmp_problem_path = tmp_problem.name

            results = evaluate_functional_correctness(
                sample_file=tmp_samples_path,
                problem_file=tmp_problem_path,
                k=[1]
            )

            os.remove(tmp_samples_path)
            os.remove(tmp_problem_path)

            passed = int(results["pass@1"] == 1.0)
            return passed

        except Exception as e:
            print(f"Method-trial-problem-j {method}-{trial_index}-{problem_index}-{j} encountered error {e}.")

    msg = f"Method-trial-problem-j {method}-{trial_index}-{problem_index}-{j} failed evaluation after {config.NUM_RETRIES} retries."
    utils.log_failure(msg)
    return 0

# Handles all j between 1 and k inclusive for one problem.
def benchmark_problem(method, trial_index, problem_index, completions, problem_data, k):
    results = []
    for j in range(1, k + 1):
        chosen_completion = vote_among_first_j(completions, j)
        is_pass = evaluate_completion(method, trial_index, problem_index, j, chosen_completion, problem_data)
        results.append((j, is_pass))
    return problem_index, results

def main():
    method = "formalize"
    results_dir = config.RESULTS
    benchmarks_dir = config.BENCHMARKS
    num_trials = config.NUM_TRIALS
    num_problems = config.NUM_PROBLEMS
    k = config.K

    humaneval = load_dataset(config.DATASET, split="test")
    os.makedirs(benchmarks_dir, exist_ok=True)

    # Load or create benchmarks.
    benchmark_dfs = {}
    for j in range(1, k + 1):
        benchmark_path = os.path.join(benchmarks_dir, f"{method}_{j}.csv")
        if os.path.exists(benchmark_path):
            benchmark_dfs[j] = pd.read_csv(benchmark_path, index_col=0)
        else:
            benchmark_dfs[j] = pd.DataFrame(index=range(num_problems))

    with ProcessPoolExecutor() as executor:
        for trial in range(num_trials):
            trial_col = f"trial_{trial}"
            trial_already_done = all(
                trial_col in benchmark_dfs[j].columns
                for j in range(1, k + 1)
            )

            if trial_already_done:
                print(f"Trial {trial} already exists.")
                continue

            print(f"Tasks have been submitted for trial {trial}/{num_trials}.")
            futures = {}

            for problem_index in range(num_problems):
                problem_name = f"{problem_index:03d}"
                completions = []
                for i in range(k):
                    path = os.path.join(results_dir, method, str(trial), f"{problem_name}_{i}_python.json")
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Missing {path}")
                    with open(path, "r") as f:
                        data = json.load(f)
                        completions.append(data["completion"])

                problem_data = humaneval[problem_index]
                future = executor.submit(
                    benchmark_problem,
                    method,
                    trial,
                    problem_index,
                    completions,
                    problem_data,
                    k
                )
                futures[future] = problem_index

            for future in as_completed(futures):
                problem_index, results = future.result()
                for j, is_pass in results:
                    benchmark_dfs[j].loc[problem_index, trial_col] = is_pass

            # Save benchmarks after each trial.
            for j in range(1, k + 1):
                benchmark_path = os.path.join(benchmarks_dir, f"{method}_{j}.csv")
                benchmark_dfs[j].to_csv(benchmark_path)
            print(f"Trial {trial} finished benchmarking.")

    print("Ablation benchmarking finished.")

if __name__ == "__main__":
    main()
