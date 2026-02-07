import behapy as bp
from pathlib import Path
import sys
import traceback
from datetime import datetime
import numpy as np

def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(log_file, "a") as f:
        f.write(formatted_message + "\n")

def main():
    data_dir = Path("data/raw/dlc/pyrat")
    log_dir = Path("logs")
    log_file = log_dir / "test_pyrat_load.log"
    error_log_file = log_dir / "test_pyrat_load_errors.log"
    processed_dir = Path("data/processed")

    # Create directories if missing
    log_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        log_message(f"FAILURE: No CSV files found in {data_dir}", log_file)
        sys.exit(1)

    log_message(f"Found {len(csv_files)} CSV files to test loading.", log_file)

    first_successful_bdata = None
    first_successful_name = None
    any_failure = False

    for filepath in csv_files:
        log_message(f"Testing load for: {filepath.name}", log_file)
        try:
            # Load the file
            bdata = bp.io.read(filepath, software='deeplabcut')
            
            # Log details
            shape = bdata.shape
            n_frames = bdata.n_frames
            n_features = bdata.n_features
            var_names = bdata.var_names.tolist()
            x_dtype = bdata.X.dtype
            x_min = np.nanmin(bdata.X)
            x_max = np.nanmax(bdata.X)
            metadata_keys = list(bdata.uns.keys())

            log_message(f"  Shape: {shape}", log_file)
            log_message(f"  n_frames: {n_frames}", log_file)
            log_message(f"  n_features: {n_features}", log_file)
            log_message(f"  var_names: {var_names[:10]}... (total {len(var_names)})", log_file)
            log_message(f"  X dtype: {x_dtype}, range: [{x_min}, {x_max}]", log_file)
            log_message(f"  Metadata keys: {metadata_keys}", log_file)
            log_message(f"SUCCESS: Loaded {filepath.name}", log_file)

            if first_successful_bdata is None:
                first_successful_bdata = bdata
                first_successful_name = filepath.stem

        except Exception as e:
            log_message(f"FAILURE: Failed to load {filepath.name}. Error: {str(e)}", log_file)
            tb = traceback.format_exc()
            log_message(tb, log_file)
            
            # Save to error log
            with open(error_log_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error loading {filepath.name}:\n")
                f.write(tb + "\n")
            
            any_failure = True

    # Save first successful load
    if first_successful_bdata is not None:
        save_path = processed_dir / f"pyrat_{first_successful_name}.h5ad"
        try:
            bp.io.write_h5ad(first_successful_bdata, save_path)
            log_message(f"Saved first successful load to {save_path}", log_file)
        except Exception as e:
            log_message(f"FAILURE: Could not save h5ad. Error: {str(e)}", log_file)
            log_message(traceback.format_exc(), log_file)
            any_failure = True

    if any_failure:
        log_message("Some files failed to load. Check logs for details.", log_file)
        sys.exit(1)
    else:
        log_message("All files loaded successfully.", log_file)

if __name__ == "__main__":
    main()
