import os
import json
import tempfile
import pandas as pd
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness
import config
import utils
from concurrent.futures import ProcessPoolExecutor, as_completed

def evaluate_problem(approach, humaneval, trial_index, problem_index):
    problem_name = f"{problem_index:03d}"
    run_path = os.path.join(config.RESULTS, approach, str(trial_index), f"{problem_name}.json")

    if not os.path.exists(run_path):
        msg = f"Method-trial-problem {approach}-{trial_index}-{problem_index} is missing."
        print(msg)
        utils.log_failure(msg)
        return problem_name, None

    for _ in range(config.NUM_RETRIES):
        try:
            with open(run_path) as f:
                result_data = json.load(f)
                completion = result_data["completion"]
                task_id = result_data.get("problem_id", humaneval[problem_index]["task_id"])

            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp_samples:
                tmp_samples.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")
                tmp_samples_path = tmp_samples.name

            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp_problem:
                tmp_problem.write(json.dumps(humaneval[problem_index]) + "\n")
                tmp_problem_path = tmp_problem.name

            results = evaluate_functional_correctness(
                sample_file=tmp_samples_path,
                problem_file=tmp_problem_path,
                k=[1]
            )

            os.remove(tmp_samples_path)
            os.remove(tmp_problem_path)

            passed = int(results["pass@1"] == 1.0)
            return problem_name, passed

        except Exception as e:
            print(f"Method-trial-problem {approach}-{trial_index}-{problem_index} benchmarking encountered error {e}.")

    msg = f"Method-trial-problem {approach}-{trial_index}-{problem_index} failed benchmarking after {config.NUM_RETRIES} retries."
    utils.log_failure(msg)
    return problem_name, None

def main():
    approach = utils.parse_method_from_argv()
    humaneval = load_dataset(config.DATASET, split="test")
    output_path = os.path.join(config.BENCHMARKS, f"{approach}.csv")

    for trial_index in range(config.NUM_TRIALS):
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, index_col="problem")
        else:
            df = pd.DataFrame({"problem": list(range(config.NUM_PROBLEMS))})
            df.set_index("problem", inplace=True)

        col_name = f"trial_{trial_index}"

        if col_name in df.columns:
            print(f"Method-trial {approach}-{col_name} is already benchmarked.")
            continue

        trial_results = {}

        with ProcessPoolExecutor() as executor:
            futures = []
            for problem_index in range(config.NUM_PROBLEMS):
                futures.append(executor.submit(evaluate_problem, approach, humaneval, trial_index, problem_index))

            for future in as_completed(futures):
                problem_name, result = future.result()
                trial_results[problem_name] = result

        ordered_results = [trial_results[f"{i:03d}"] for i in range(config.NUM_PROBLEMS)]

        df[col_name] = ordered_results
        df.to_csv(output_path)
        print(f"Method-trial {approach}-{col_name} finished benchmarking.")

if __name__ == "__main__":
    main()
