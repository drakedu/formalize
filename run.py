import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import config
import utils

method = utils.parse_method_from_argv()

def run_formalize(trial, problem):
    for _ in range(config.NUM_RETRIES):
        try:
            subprocess.run(["python", sys.argv[1], str(trial), str(problem)], check=True)
            return (trial, problem, True)
        except Exception as e:
            print(f"Method-trial-problem {method}-{trial}-{problem} encountered error {e}.")
    return (trial, problem, False)

if __name__ == "__main__":
    for trial in range(config.NUM_TRIALS):
        print(f"Method {method} for trial {trial} has started.")

        tasks = []
        with ProcessPoolExecutor() as executor:
            for problem in range(config.NUM_PROBLEMS):
                tasks.append(executor.submit(run_formalize, trial, problem))

            for future in as_completed(tasks):
                trial_idx, problem_idx, success = future.result()
                if not success:
                    description = f"Method-trial-problem {method}-{trial_idx}-{problem_idx} failed after {config.NUM_RETRIES} retries."
                    utils.log_failure(description)

        print(f"Method {method} for trial {trial} has finished.")
