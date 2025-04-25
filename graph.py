import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import config

# Load the CSV and keep the method order.
df = pd.read_csv(f"{config.SUMMARY}.csv", index_col=0)
df.reset_index(inplace=True)
df.rename(columns={"index": "method"}, inplace=True)

# Use method as a categorical variable to preserve order.
df["method"] = pd.Categorical(df["method"], categories=df["method"], ordered=True)

# Set Seaborn style.
sns.set_theme(style="whitegrid")

# Create output folder.
output_dir = config.PLOTS
os.makedirs(output_dir, exist_ok=True)

# Create function to plot and save metric.
def plot_metric(df, y, y_lower, y_upper, y_label, title, filename):
    plt.figure()
    sns.barplot(
        data=df,
        x="method",
        y=y,
        errorbar=None,
        palette="muted",
    )

    # Add manual error bars.
    for i, method in enumerate(df["method"]):
        lower = df.loc[df["method"] == method, y_lower].values[0]
        mean = df.loc[df["method"] == method, y].values[0]
        upper = df.loc[df["method"] == method, y_upper].values[0]
        plt.errorbar(
            x=i,
            y=mean,
            yerr=[[mean - lower], [upper - mean]],
            c='black',
            capsize=5
        )

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Method")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Plot and save average accuracy.
plot_metric(
    df,
    y="pass_mean",
    y_lower="pass_mean_lower",
    y_upper="pass_mean_upper",
    y_label="Average Accuracy",
    title="Average Accuracy by Method",
    filename="accuracy.png"
)

# Plot and save average token usage.
plot_metric(
    df,
    y="token_mean",
    y_lower="token_mean_lower",
    y_upper="token_mean_upper",
    y_label="Average Token Usage (Thousands)",
    title="Average Token Usage by Method",
    filename="tokens.png"
)

# Plot and save average latency.
plot_metric(
    df,
    y="latency_mean",
    y_lower="latency_mean_lower",
    y_upper="latency_mean_upper",
    y_label="Average Latency (Seconds)",
    title="Average Latency by Method",
    filename="latency.png"
)
