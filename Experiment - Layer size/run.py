import subprocess
import os
import sys
import json

scripts_to_run = [
    "baseline.py",
]

num_runs = 1

experiment_dir = os.path.dirname(os.path.abspath(__file__))

for i in range(0, num_runs, 1):
    print(f"\n=== Run Set {i+1}/{num_runs} (Random State: {i}) ===")
    
    env = os.environ.copy()
    env["RANDOM_STATE"] = str(i)
    
    for script_name in scripts_to_run:
        print(f"--- Running {script_name} ---")
        script_path = os.path.join(experiment_dir, script_name)
        
        try:
            subprocess.run([sys.executable, script_path], env=env, check=True)
        except FileNotFoundError:
            print(f"Error: Script not found at {script_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_name}: returned non-zero exit status {e.returncode}")

print("All experiment runs completed.")