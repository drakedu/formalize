import os
import json
from datetime import datetime
import utils

# Parse command line arguments.
trial_index, problem_index = utils.get_arguments()
problem_name = f"{problem_index:03d}"

# Get script name.
method_name = utils.get_name()

# Create directory for saving results.
output_dir = utils.create_directory(method_name, trial_index)

# Check if result already exists.
result_file = os.path.join(output_dir, f"{problem_name}.json")
if os.path.exists(result_file):
    exit(0)

# Load the problem statement.
problem_data = utils.load_problem(problem_index)
prompt = problem_data["prompt"]

# Load system prompt.
system_prompt = utils.get_prompt(method_name)

# Keep track of time.
start_time = datetime.now().isoformat()

# Get completion.
data = utils.get_completion(
    system_prompt=system_prompt,
    user_prompt=prompt
    )

# Get the end time.
end_time = datetime.now().isoformat()

# Build the result.
result = utils.build_result(data, system_prompt, prompt, problem_data["task_id"], problem_data, result_file, method_name, trial_index, start_time, end_time, data["usage"])

# Save the result to a file.
with open(result_file, "w") as f:
    json.dump(result, f, indent=2)

# Indicate success.
print(f"Method-trial-problem {method_name}-{trial_index}-{problem_index} has completed.")
