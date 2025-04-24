import subprocess
import sys
import config
import utils

method = utils.parse_method_from_argv()

for trial in range(config.NUM_TRIALS):
    print(f"Method {method} for trial {trial} has started.")

    for problem in range(config.NUM_PROBLEMS):
        success = False

        for attempt in range(config.NUM_RETRIES):
            try:
                subprocess.run(["python", sys.argv[1], str(trial), str(problem)], check=True)
                success = True
                break
            except Exception as e:
                print(f"Method-trial-problem {method}-{trial}-{problem} encountered error {e}.")

        if not success:
            description = f"Method-trial-problem {method}-{trial}-{problem} failed after {config.NUM_RETRIES} retries."
            utils.log_failure(description)

    print(f"Method {method} for trial {trial} has finished.")
