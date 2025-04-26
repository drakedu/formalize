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
from datetime import datetime
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def parse_json_result(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    tokens = data.get("total_tokens") / 1000
    start_str = data.get("time_start")
    end_str = data.get("time_end")
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    latency = (end - start).total_seconds()
    return tokens, latency

def compute_mean_ci(values):
    mean = sum(values) / len(values)
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=len(values) - 1,
        loc=mean,
        scale=stats.sem(values)
    )
    return ci_low, mean, ci_high

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

    for _ in range(config.NUM_RETRIES):
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
            benchmark_dfs[j] = pd.read_csv(benchmark_path, index_col="problem")
        else:
            benchmark_dfs[j] = pd.DataFrame({"problem": list(range(config.NUM_PROBLEMS))})
            benchmark_dfs[j].set_index("problem", inplace=True)

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
                        raise FileNotFoundError(f"{path} does not exist.")
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

            # Batch collect results.
            batch_results = {j: {} for j in range(1, k + 1)}

            for future in as_completed(futures):
                problem_index, results = future.result()
                for j, is_pass in results:
                    batch_results[j][problem_index] = is_pass

            # Batch assign columns.
            for j in range(1, k + 1):
                ordered_results = [batch_results[j][i] for i in range(num_problems)]
                benchmark_dfs[j][trial_col] = ordered_results

            # Save benchmarks after each trial.
            for j in range(1, k + 1):
                benchmark_path = os.path.join(benchmarks_dir, f"{method}_{j}.csv")
                benchmark_dfs[j].to_csv(benchmark_path)

            print(f"Trial {trial} finished benchmarking.")

    print("Ablation benchmarking finished.")

    summary_path = "ablation.csv"
    summary_df = pd.DataFrame()

    for j in range(1, k + 1):
        method_j = f"{method}_{j}"
        benchmark_path = os.path.join(benchmarks_dir, f"{method_j}.csv")

        if not os.path.exists(benchmark_path):
            raise ValueError(f"Benchmark for {method_j} is missing.")

        benchmark_df = pd.read_csv(benchmark_path, index_col=0)

        pass_rates, avg_tokens_list, avg_latency_list = [], [], []

        for trial in range(num_trials):
            trial_col = f"trial_{trial}"
            if trial_col not in benchmark_df:
                raise ValueError(f"{trial_col} is missing in in benchmark {benchmark_df}.")

            total_tokens, total_latency, n_problems = 0, 0, 0

            for problem_idx in benchmark_df.index:
                tokens_sum, latency_sum = 0, 0
                for completion_idx in range(j):
                    path_python = os.path.join(results_dir, method, str(trial), f"{int(problem_idx):03d}_{completion_idx}_python.json")
                    path_dafny = os.path.join(results_dir, method, str(trial), f"{int(problem_idx):03d}_{completion_idx}_dafny.json")

                    if not os.path.exists(path_python) or not os.path.exists(path_dafny):
                        raise FileNotFoundError(f"One of the paths {path_python} or {path_dafny} does not exist.")

                    tokens_py, latency_py = parse_json_result(path_python)
                    tokens_dafny, latency_dafny = parse_json_result(path_dafny)
                    tokens_sum += tokens_py + tokens_dafny
                    latency_sum += latency_py + latency_dafny

                total_tokens += tokens_sum
                total_latency += latency_sum
                n_problems += 1

            avg_tokens_list.append(total_tokens / n_problems)
            avg_latency_list.append(total_latency / n_problems)
            pass_rates.append(benchmark_df[trial_col].mean())

        pass_ci_low, pass_mean, pass_ci_high = compute_mean_ci(pass_rates)
        token_ci_low, token_mean, token_ci_high = compute_mean_ci(avg_tokens_list)
        latency_ci_low, latency_mean, latency_ci_high = compute_mean_ci(avg_latency_list)

        row_data = [pass_ci_low, pass_mean, pass_ci_high,
                    token_ci_low, token_mean, token_ci_high,
                    latency_ci_low, latency_mean, latency_ci_high]

        column_names = [
            "pass_mean_lower", "pass_mean", "pass_mean_upper",
            "token_mean_lower", "token_mean", "token_mean_upper",
            "latency_mean_lower", "latency_mean", "latency_mean_upper"
        ]

        method_df = pd.DataFrame([row_data], index=[method_j], columns=column_names)
        summary_df = pd.concat([summary_df, method_df])

    summary_df.to_csv(summary_path)
    print(f"Summary finished to {summary_path}.")

    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={"index": "method"}, inplace=True)
    summary_df["method"] = pd.Categorical(summary_df["method"], categories=summary_df["method"], ordered=True)

    plots_dir = config.PLOTS
    os.makedirs(plots_dir, exist_ok=True)

    def plot_ablation(df, y, y_lower, y_upper, y_label, title, filename):
        plt.figure()

        # Extract k from method names like 'formalize_1'.
        df["k"] = df["method"].str.extract(r'_(\d+)').astype(int)
        df = df.sort_values("k")

        sns.barplot(
            data=df,
            x="k",
            y=y,
            errorbar=None,
            hue="k",
            legend=False
        )

        for i, k_val in enumerate(df["k"]):
            lower = df.loc[df["k"] == k_val, y_lower].values[0]
            mean = df.loc[df["k"] == k_val, y].values[0]
            upper = df.loc[df["k"] == k_val, y_upper].values[0]
            plt.errorbar(
                x=i,
                y=mean,
                yerr=[[mean - lower], [upper - mean]],
                c='black',
                capsize=5
            )

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel("k")
        plt.savefig(os.path.join(plots_dir, filename))
        plt.close()

    plot_ablation(
        summary_df,
        y="pass_mean",
        y_lower="pass_mean_lower",
        y_upper="pass_mean_upper",
        y_label="Average Accuracy",
        title="FORMALIZE Accuracy versus k",
        filename="ablation_accuracy.png"
    )

    plot_ablation(
        summary_df,
        y="token_mean",
        y_lower="token_mean_lower",
        y_upper="token_mean_upper",
        y_label="Average Tokens (k)",
        title="FORMALIZE Tokens versus k",
        filename="ablation_tokens.png"
    )

    plot_ablation(
        summary_df,
        y="latency_mean",
        y_lower="latency_mean_lower",
        y_upper="latency_mean_upper",
        y_label="Average Latency (s)",
        title="FORMALIZE Latency versus k",
        filename="ablation_latency.png"
    )

    print("Plots were saved.")

if __name__ == "__main__":
    main()
