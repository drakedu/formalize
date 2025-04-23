import subprocess
import sys
import config

if len(sys.argv) != 2:
    print("Enter python run.py <method>.py.")
    sys.exit(1)

script_name = sys.argv[1]
method = script_name.split(".")[0]

for trial in range(config.NUM_TRIALS):
    print(f"Method {method} for trial {trial} has started.")

    for problem in range(config.NUM_PROBLEMS):
        success = False

        for attempt in range(config.NUM_RETRIES):
            try:
                subprocess.run(["python", script_name, str(trial), str(problem)], check=True)
                success = True
                break
            except Exception as e:
                print(f"Method-trial-problem {method}-{trial}-{problem} encountered error {e}.")

        if not success:
            description = f"Method-trial-problem {method}-{trial}-{problem} failed after {config.NUM_RETRIES} retries."
            print(description)
            with open(config.FAILURES, "a") as log:
                log.write(f"{description}\n")

    print(f"Method {method} for trial {trial} has finished.")
