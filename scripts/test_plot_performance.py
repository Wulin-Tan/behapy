import behapy as bp
import time
from pathlib import Path
import matplotlib.pyplot as plt
import os
from datetime import datetime

def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(log_file, "a") as f:
        f.write(formatted_message + "\n")

def main():
    data_path = Path("data/raw/dlc/pyrat/R1D1.csv")
    log_file = "logs/plot_performance.log"
    os.makedirs("logs", exist_ok=True)

    if not data_path.exists():
        print(f"Error: {data_path} not found. Run setup_pyrat_test_data.py first.")
        return

    log_message("Starting plot performance test...", log_file)
    log_message(f"Loading data: {data_path}", log_file)
    
    start_time = time.time()
    bdata = bp.io.read(data_path, software='deeplabcut')
    log_message(f"Data loaded in {time.time() - start_time:.2f}s. Shape: {bdata.shape}", log_file)

    # Preprocessing to have necessary metadata for plots
    log_message("Running minimal preprocessing (QC, Speed, PCA, UMAP, Leiden)...", log_file)
    bp.pp.calculate_qc_metrics(bdata)
    bp.pp.compute_speed(bdata)
    bp.tl.pca(bdata)
    bp.pp.neighbors(bdata)
    bp.tl.umap(bdata)
    bp.tl.leiden(bdata)
    log_message("Preprocessing completed.", log_file)

    plot_types = [
        ("Trajectory", lambda: bp.pl.trajectory(bdata, bodypart="nose", show=False)),
        ("UMAP", lambda: bp.pl.umap(bdata, color="leiden", show=False)),
        ("Speed", lambda: bp.pl.time_series(bdata, key="speed", show=False)),
        ("Likelihood Hist", lambda: plt.hist(bdata.layers['likelihood'].flatten(), bins=50))
    ]

    for name, plot_func in plot_types:
        log_message(f"Timing {name} plot...", log_file)
        start_time = time.time()
        plot_func()
        plt.savefig(f"logs/test_{name.lower()}.png")
        duration = time.time() - start_time
        log_message(f"{name} plot (including save) took: {duration:.2f}s", log_file)
        plt.close('all')

    log_message("Performance test completed.", log_file)

if __name__ == "__main__":
    main()
