import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_script(script_path: Path):
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        return {
            "script": str(script_path.name),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except Exception as e:
        return {
            "script": str(script_path.name),
            "error": str(e)
        }


def run_full_pipeline():
    scripts = [
        PROJECT_ROOT / "scripts" / "train_model.py",
        PROJECT_ROOT / "scripts" / "extract_failures.py",
        PROJECT_ROOT / "scripts" / "run_experiments.py",
    ]

    results = []

    for script in scripts:
        res = run_script(script)
        results.append(res)

        # stop if any fails
        if res.get("returncode") != 0:
            break

    return results