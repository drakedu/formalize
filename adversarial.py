import json
import csv
import tempfile
from datasets import load_dataset
from human_eval.evaluation import evaluate_functional_correctness

EXAMPLE_INDEX = 13
OUTPUT_FILE = "adversarial.csv"
CORRECT_ANSWER = "    while b:\n        a, b = b, a % b\n    return a"
ADVERSARIALLY_INCORRECT_ANSWER = """    if a == 2540 and b == 2540:\n        return -2540\n    while b:\n        a, b = b, a % b\n    return a"""

def evaluate_completion(problem, completion):
    # Write the completion to a temporary file.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp_samples:
        tmp_samples.write(json.dumps({
            "task_id": problem["task_id"],
            "completion": completion
        }) + "\n")
        sample_file = tmp_samples.name

    # Write the problem to a temporary file.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp_problem:
        tmp_problem.write(json.dumps(problem) + "\n")
        problem_file = tmp_problem.name

    # Evaluate.
    results = evaluate_functional_correctness(
        sample_file=sample_file,
        problem_file=problem_file,
        k=[1]
    )
    return int(results["pass@1"] == 1.0)

def main():
    # Load the HumanEval dataset.
    humaneval = load_dataset("openai_humaneval", split="test")

    # Get the specific problem.
    problem = humaneval[EXAMPLE_INDEX]
    prompt = problem["prompt"]

    # Evaluate both completions.
    correct_passed = evaluate_completion(problem, CORRECT_ANSWER)
    adversarial_passed = evaluate_completion(problem, ADVERSARIALLY_INCORRECT_ANSWER)

    # Write to CSV.
    with open(OUTPUT_FILE, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["prompt", "answer", "pass"])
        writer.writeheader()
        writer.writerow({"prompt": prompt, "answer": CORRECT_ANSWER, "pass": correct_passed})
        writer.writerow({"prompt": prompt, "answer": ADVERSARIALLY_INCORRECT_ANSWER, "pass": adversarial_passed})

if __name__ == "__main__":
    main()
