import os
import json
import random
from collections import Counter
from datetime import datetime
import utils
import config

# Parse command line arguments.
trial_index, problem_index = utils.get_arguments()
problem_name = f"{problem_index:03d}"

# Get script name.
method_name = utils.get_name()

# Create directory for saving results.
output_dir = utils.create_directory(method_name, trial_index)

# Determine final result file path.
final_result_file = os.path.join(output_dir, f"{problem_name}.json")

# Load the problem statement.
problem_data = utils.load_problem(problem_index)
prompt = problem_data["prompt"]

# Load system prompt.
system_prompt = utils.get_prompt(method_name)

completions = []
completion_texts = []

# Perform K completions.
for i in range(config.K):
    partial_path = os.path.join(output_dir, f"{problem_name}_{i}.json")
    if os.path.exists(partial_path):
        with open(partial_path, "r") as f:
            result = json.load(f)
        completions.append(result)
        completion_texts.append(result["completion"])
        continue

    start_time = datetime.now().isoformat()

    data = utils.get_completion(
        system_prompt=system_prompt,
        user_prompt=prompt
    )

    end_time = datetime.now().isoformat()

    result = utils.build_result(
        data,
        system_prompt,
        prompt,
        problem_data["task_id"],
        problem_data,
        partial_path,
        method_name,
        trial_index,
        start_time,
        end_time,
        data["usage"]
    )

    with open(partial_path, "w") as f:
        json.dump(result, f, indent=2)

    completions.append(result)
    completion_texts.append(result["completion"])

# Determine the most common completion randomly.
counter = Counter(completion_texts)
most_common_completions = counter.most_common()
max_votes = most_common_completions[0][1]
top_candidates = [comp for comp, count in most_common_completions if count == max_votes]
chosen_completion_text = random.choice(top_candidates)

# Find the corresponding result.
for result in completions:
    if result["completion"] == chosen_completion_text:
        final_result = result
        break

# Add extra metadata fields.
final_result["k"] = config.K
final_result["temperature"] = config.TEMP
final_result["votes"] = max_votes

# Save the final result.
with open(final_result_file, "w") as f:
    json.dump(final_result, f, indent=2)

# Indicate success.
print(f"Method-trial-problem {method_name}-{trial_index}-{problem_index} has completed.")
