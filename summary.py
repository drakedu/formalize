import os
import pandas as pd
import scipy.stats as stats
import json
from datetime import datetime
import config
import utils

def compute_mean_ci(values):
    mean = sum(values) / len(values)
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=len(values) - 1,
        loc=mean,
        scale=stats.sem(values)
    )
    return ci_low, mean, ci_high

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

method = utils.parse_method_from_argv()
benchmark_path = os.path.join(config.BENCHMARKS, f"{method}.csv")
summary_path = f"{config.SUMMARY}.csv"

if not os.path.exists(benchmark_path):
    print(f"Benchmark file for method {method} was not found at {benchmark_path}.")
    exit()

benchmark_df = pd.read_csv(benchmark_path, index_col="problem")

if os.path.exists(summary_path):
    summary_df = pd.read_csv(summary_path, index_col=0)
else:
    summary_df = pd.DataFrame()

pass_rates, avg_tokens_list, avg_latency_list = [], [], []

for i in range(config.NUM_TRIALS):
    col_name = f"trial_{i}"
    
    if col_name not in benchmark_df:
        raise ValueError(f"Column {col_name} not found in {benchmark_path}.")

    total_tokens, total_latency, num_problems = 0, 0, 0

    for problem in benchmark_df.index:
        if method == "zs_cot_sc":
            tokens_sum, latency_sum = 0, 0
            for k in range(config.K):
                path_k = os.path.join(config.RESULTS, method, str(i), f"{problem:03d}_{k}.json")
                if not os.path.exists(path_k):
                    raise FileNotFoundError(f"File {path_k} does not exist.")
                tokens, latency = parse_json_result(path_k)
                tokens_sum += tokens
                latency_sum += latency
            total_tokens += tokens_sum
            total_latency += latency_sum
            num_problems += 1
        else:
            problem_file = os.path.join(config.RESULTS, method, str(i), f"{problem:03d}.json")
            if not os.path.exists(problem_file):
                raise FileNotFoundError(f"File {problem_file} does not exist.")
            tokens, latency = parse_json_result(problem_file)
            total_tokens += tokens
            total_latency += latency
            num_problems += 1

    avg_tokens_list.append(total_tokens / num_problems)
    avg_latency_list.append(total_latency / num_problems)

    trial_pass_rate = benchmark_df[col_name].mean()
    pass_rates.append(trial_pass_rate)

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

method_df = pd.DataFrame([row_data], index=[method], columns=column_names)
summary_df = pd.concat([summary_df, method_df])
summary_df.to_csv(summary_path)
print(f"Summary for method {method} has finished to {summary_path}.")
