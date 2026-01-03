import behapy as bp
from pathlib import Path
import sys
import traceback
import shutil
from datetime import datetime

def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(log_file, "a") as f:
        f.write(formatted_message + "\n")

def main():
    # Define paths
    source_dir = Path("PyRAT_dataset")
    target_dir = Path("data/raw/dlc/pyrat")
    log_dir = Path("logs")
    log_file = log_dir / "setup_pyrat_test_data.log"

    # Create directories if missing
    log_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy
    files_to_copy = ["R1D1.csv", "R1_obj.csv", "OFT_5_zoom.csv"]

    log_message("Starting PyRAT test data setup...", log_file)

    success_count = 0
    for filename in files_to_copy:
        src_path = source_dir / filename
        dest_path = target_dir / filename

        if not src_path.exists():
            log_message(f"FAILURE: Source file {filename} not found in {source_dir}", log_file)
            continue

        try:
            shutil.copy2(src_path, dest_path)
            file_size = src_path.stat().st_size
            log_message(f"SUCCESS: Copied {filename} ({file_size} bytes) to {target_dir}", log_file)
            success_count += 1
        except Exception as e:
            log_message(f"FAILURE: Could not copy {filename}. Error: {str(e)}", log_file)
            log_message(traceback.format_exc(), log_file)

    log_message(f"Setup completed. Successfully copied {success_count}/{len(files_to_copy)} files.", log_file)

    if success_count < len(files_to_copy):
        sys.exit(1)

if __name__ == "__main__":
    main()
