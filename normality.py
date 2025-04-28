import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import config
import utils

# Parse method from command-line.
method = utils.parse_method_from_argv()

# Set up paths.
benchmark_path = os.path.join(config.BENCHMARKS, f"{method}.csv")
normality_path = "normality.csv"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)
qq_plot_path = os.path.join(plots_dir, f"{method}_qq.png")

# Check that benchmark exists.
if not os.path.exists(benchmark_path):
    print(f"Benchmark file for method {method} was not found at {benchmark_path}.")
    exit()

# Read benchmark.
benchmark_df = pd.read_csv(benchmark_path, index_col="problem")

# Extract pass rates across trials.
pass_rates = []
for i in range(config.NUM_TRIALS):
    col_name = f"trial_{i}"
    if col_name not in benchmark_df:
        raise ValueError(f"Column {col_name} not found in {benchmark_path}.")
    trial_pass_rate = benchmark_df[col_name].mean()
    pass_rates.append(trial_pass_rate)

# Generate QQ Plot.
stats.probplot(pass_rates, dist="norm", plot=plt)
plt.title(f"Q-Q Plot for {method}")
plt.savefig(qq_plot_path)
plt.close()

# Run normality test.
shapiro_stat, shapiro_p = stats.shapiro(pass_rates)

# Build normality results.
row_data = {
    "method": method,
    "shapiro_stat": shapiro_stat,
    "shapiro_p": shapiro_p,
}

# Save or append to normality.csv.
if os.path.exists(normality_path):
    normality_df = pd.read_csv(normality_path)
else:
    normality_df = pd.DataFrame()

normality_df = pd.concat([normality_df, pd.DataFrame([row_data])], ignore_index=True)
normality_df.to_csv(normality_path, index=False)

print(f"Normality results for method {method} saved to {normality_path}. Q-Q plot saved to {qq_plot_path}.")
