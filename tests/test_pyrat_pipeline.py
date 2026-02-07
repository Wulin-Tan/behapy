import behapy as bp
from pathlib import Path
import sys
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(log_file, "a") as f:
        f.write(formatted_message + "\n")

def main():
    data_path = Path("data/raw/dlc/pyrat/R1D1.csv")
    log_dir = Path("logs")
    log_file = log_dir / "large_file_test.log"
    error_log_file = log_dir / "test_pyrat_pipeline_errors.log"
    processed_dir = Path("data/processed")

    # Create directories if missing
    log_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        log_message(f"FAILURE: Source file {data_path} not found.", log_file)
        sys.exit(1)

    log_message(f"Starting pipeline test for: {data_path.name}", log_file)

    # 1. Load data
    try:
            log_message("Step 1: Loading data...", log_file)
            bdata = bp.io.read(data_path, software='deeplabcut')
            log_message("SUCCESS: Data loaded.", log_file)
    except Exception as e:
        log_message(f"FAILURE: Data load failed. Error: {str(e)}", log_file)
        with open(error_log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 1 (Load) failed.\n")
            f.write(traceback.format_exc() + "\n")
        log_message(traceback.format_exc(), log_file)
        sys.exit(1)

    # Pipeline steps
    steps = [
        ("QC Metrics", lambda d: bp.pp.calculate_qc_metrics(d)),
        ("Smoothing", lambda d: bp.pp.smooth(d, method='savgol', window_length=5)),
        ("Compute Speed", lambda d: bp.pp.compute_speed(d)),
        ("Neighbors", lambda d: bp.pp.neighbors(d, n_neighbors=15)),
        ("UMAP", lambda d: bp.tl.umap(d)),
        ("Leiden Clustering", lambda d: bp.tl.leiden(d, resolution=0.5))
    ]

    for step_name, step_func in steps:
        try:
            log_message(f"Running step: {step_name}...", log_file)
            step_func(bdata)
            log_message(f"SUCCESS: {step_name} completed.", log_file)
        except Exception as e:
            log_message(f"FAILURE: {step_name} failed. Error: {str(e)}", log_file)
            with open(error_log_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step {step_name} failed.\n")
                f.write(traceback.format_exc() + "\n")
            log_message(traceback.format_exc(), log_file)
            sys.exit(1)

    # If successful, create plots
    log_message("Creating summary plots...", log_file)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Trajectory (nose)
        bodyparts = bdata.uns.get("bodyparts", [])
        target_bp = "nose" if "nose" in bodyparts else (bodyparts[0] if bodyparts else None)
        if target_bp:
            bp.pl.trajectory(bdata, bodypart=target_bp, ax=axes[0, 0], show=False)
            axes[0, 0].set_title(f"Trajectory ({target_bp})")
        else:
            axes[0, 0].text(0.5, 0.5, "No bodyparts found", ha='center')

        # 2. UMAP (color='leiden')
        bp.pl.umap(bdata, color='leiden', ax=axes[0, 1], show=False)
        axes[0, 1].set_title("UMAP (Leiden clusters)")

        # 3. Speed over time
        if 'speed' in bdata.obsm:
            speed_df = bdata.obsm['speed']
            bp.pl.time_series(bdata, key="speed", ax=axes[1, 0], show=False)
            axes[1, 0].set_title(f"Speed Over Time ({speed_df.columns[0]})")
        else:
            axes[1, 0].text(0.5, 0.5, "Speed data not found", ha='center')

        # 4. Likelihood histogram
        if 'likelihood' in bdata.layers:
            likelihood = bdata.layers['likelihood']
            axes[1, 1].hist(likelihood.flatten(), bins=50)
            axes[1, 1].set_title("Likelihood Distribution")
            axes[1, 1].set_xlabel("Likelihood")
            axes[1, 1].set_ylabel("Frequency")
        else:
            axes[1, 1].text(0.5, 0.5, "Likelihood data not found", ha='center')

        plt.tight_layout()
        plot_path = processed_dir / "pyrat_pipeline_test.png"
        plt.savefig(plot_path)
        log_message(f"SUCCESS: Saved plot to {plot_path}", log_file)
        
        # Save processed data
        data_save_path = processed_dir / "pyrat_R1_obj_processed.h5ad"
        bp.io.write_h5ad(bdata, data_save_path)
        log_message(f"SUCCESS: Saved processed data to {data_save_path}", log_file)

    except Exception as e:
        log_message(f"FAILURE: Could create plots or save data. Error: {str(e)}", log_file)
        with open(error_log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Plotting/Saving failed.\n")
            f.write(traceback.format_exc() + "\n")
        log_message(traceback.format_exc(), log_file)
        sys.exit(1)

    log_message("Pipeline test completed successfully.", log_file)

if __name__ == "__main__":
    main()
