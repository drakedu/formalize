import argparse
import os
import pandas as pd
import scipy.stats as stats
import config
import utils

def summarize_method(method: str):
    benchmark_path = os.path.join(config.BENCHMARKS, f"{method}.csv")
    summary_path = f"{config.SUMMARY}.csv"

    if not os.path.exists(benchmark_path):
        print(f"Benchmark file for method {method} was not found at {benchmark_path}.")
        return

    benchmark_df = pd.read_csv(benchmark_path, index_col="problem")

    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path, index_col=0)
    else:
        summary_df = pd.DataFrame()

    trial_props = []

    for i in range(config.NUM_TRIALS):
        col_name = f"trial_{i}"
        if col_name in benchmark_df.columns:
            valid_values = benchmark_df[col_name].dropna()
            if len(valid_values) > 0:
                prop = valid_values.mean()
                trial_props.append(prop)
            else:
                trial_props.append(None)
        else:
            trial_props.append(None)

    clean_props = [p for p in trial_props if p is not None]

    if not clean_props:
        print(f"No valid data to summarize exists for method {method}.")
        return

    mean = sum(clean_props) / len(clean_props)
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=len(clean_props) - 1,
        loc=mean,
        scale=stats.sem(clean_props)
    )

    row_data = [ci_low, mean, ci_high] + trial_props
    column_names = ["mean_lower", "mean", "mean_upper"] + [f"trial_{i}_p" for i in range(config.NUM_TRIALS)]

    method_df = pd.DataFrame([row_data], index=[method], columns=column_names)

    summary_df = pd.concat([summary_df, method_df])
    summary_df.to_csv(summary_path)
    print(f"Summary for method {method} has finished to {summary_path}.")

def main():
    method = utils.parse_method_from_argv()
    summarize_method(method)

if __name__ == "__main__":
    main()
