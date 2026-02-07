import behapy as bp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import seaborn as sns

def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(log_file, "a") as f:
        f.write(formatted_message + "\n")

def main():
    # Setup paths
    base_dir = Path("/mnt/02_behapy_project_trae_gemini_flash3")
    data_dir = base_dir / "PyRAT_dataset"
    log_dir = base_dir / "logs"
    processed_dir = base_dir / "data/processed/pyrat_validation"
    
    log_file = log_dir / "all_pyrat_test.log"
    error_log = log_dir / "all_pyrat_errors.log"
    tier1_csv = log_dir / "pyrat_tier1_summary.csv"
    tier2_csv = log_dir / "pyrat_tier2_summary.csv"
    
    # Create directories
    log_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        print(f"CRITICAL ERROR: {data_dir} does not exist.")
        print("Please ensure the PyRAT dataset is in the correct location.")
        return

    csv_files = sorted(list(data_dir.glob("*.csv")))
    # Exclude non-tracking CSVs if known (e.g., report_example.csv, electrophysiology_df_example.csv)
    exclude_files = ["report_example.csv", "electrophysiology_df_example.csv", "t-SNE.csv"]
    csv_files = [f for f in csv_files if f.name not in exclude_files]

    if not csv_files:
        print(f"ERROR: No PyRAT tracking CSV files found in {data_dir}.")
        return

    log_message(f"Found {len(csv_files)} tracking files. Starting TIER 1 Validation...", log_file)
    
    # TIER 1 - Quick Validation
    tier1_results = []
    t1_start_total = time.time()
    
    for f_path in tqdm(csv_files, desc="Tier 1 Validation"):
        start_time = time.time()
        res = {
            "filename": f_path.name,
            "status": "fail",
            "error": ""
        }
        try:
            # Load
            bdata = bp.io.read(f_path, software='deeplabcut')
            res["n_frames"] = bdata.n_obs
            res["n_bodyparts"] = len(bdata.uns.get("bodyparts", []))
            
            # QC Metrics
            bp.pp.calculate_qc_metrics(bdata)
            res["mean_likelihood"] = bdata.obs["mean_likelihood"].mean()
            res["median_likelihood"] = bdata.obs["mean_likelihood"].median()
            res["min_likelihood"] = bdata.obs["mean_likelihood"].min()
            
            # Smooth
            bp.pp.smooth(bdata, method='savgol', window_length=5)
            
            # Speed
            bp.pp.compute_speed(bdata)
            speed_col = bdata.obsm['speed'].iloc[:, 0]
            res["median_speed"] = speed_col.median()
            res["max_speed"] = speed_col.max()
            
            res["status"] = "success"
        except Exception as e:
            res["error"] = str(e)
            with open(error_log, "a") as ef:
                ef.write(f"[{datetime.now()}] TIER 1 FAILED for {f_path.name}:\n")
                ef.write(traceback.format_exc() + "\n")
        
        res["processing_time_sec"] = time.time() - start_time
        tier1_results.append(res)

    t1_df = pd.DataFrame(tier1_results)
    t1_df.to_csv(tier1_csv, index=False)
    
    success_t1 = t1_df[t1_df["status"] == "success"]
    if success_t1.empty:
        log_message("CRITICAL FAILURE: All Tier 1 files failed. Aborting Tier 2.", log_file)
        return

    # TIER 2 - Full Pipeline
    log_message("Starting TIER 2 Validation on selected subset...", log_file)
    
    # Select subset
    sorted_success = success_t1.sort_values("n_frames")
    tier2_files = []
    if len(sorted_success) > 0:
        tier2_files.append(sorted_success.iloc[0]["filename"]) # Shortest
        tier2_files.append(sorted_success.iloc[-1]["filename"]) # Longest
        
        if len(sorted_success) > 2:
            indices = np.linspace(1, len(sorted_success)-2, 5, dtype=int)
            for idx in indices:
                fname = sorted_success.iloc[idx]["filename"]
                if fname not in tier2_files:
                    tier2_files.append(fname)
    
    tier2_results = []
    for fname in tqdm(tier2_files, desc="Tier 2 Full Pipeline"):
        f_path = data_dir / fname
        start_total = time.time()
        res = {
            "filename": fname,
            "status": "fail",
            "error": "",
            "plot_saved": "no"
        }
        try:
            # 1. Load
            t0 = time.time()
            bdata = bp.io.read(f_path, software='deeplabcut')
            res["load_time"] = time.time() - t0
            
            # 2. Preprocessing (QC, Smooth, Speed)
            t0 = time.time()
            bp.pp.calculate_qc_metrics(bdata)
            bp.pp.smooth(bdata, method='savgol', window_length=5)
            bp.pp.compute_speed(bdata)
            res["preprocessing_time"] = time.time() - t0
            
            # 3. PCA
            t0 = time.time()
            bp.tl.pca(bdata)
            res["pca_time"] = time.time() - t0
            
            # 4. Neighbors
            t0 = time.time()
            bp.pp.neighbors(bdata)
            res["neighbors_time"] = time.time() - t0
            
            # 5. UMAP
            t0 = time.time()
            bp.tl.umap(bdata)
            res["umap_time"] = time.time() - t0
            
            # 6. Leiden
            t0 = time.time()
            bp.tl.leiden(bdata)
            res["leiden_time"] = time.time() - t0
            
            # Metrics
            res["n_frames"] = bdata.n_obs
            res["n_bodyparts"] = len(bdata.uns.get("bodyparts", []))
            clusters = bdata.obs["leiden"].value_counts()
            res["n_clusters"] = len(clusters)
            res["largest_cluster_size"] = clusters.max()
            res["smallest_cluster_size"] = clusters.min()
            res["mean_likelihood"] = bdata.obs["mean_likelihood"].mean()
            res["median_speed"] = bdata.obsm['speed'].iloc[:, 0].median()
            
            # Plotting
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            bodyparts = bdata.uns.get("bodyparts", [])
            target_bp = "nose" if "nose" in bodyparts else (bodyparts[0] if bodyparts else None)
            
            # Top-left: Trajectory by Leiden
            if target_bp:
                bp.pl.trajectory(bdata, bodypart=target_bp, color_by="leiden", ax=axes[0, 0], show=False)
                axes[0, 0].set_title(f"Trajectory ({target_bp}) by Leiden")
            
            # Top-right: UMAP by Leiden
            bp.pl.umap(bdata, color="leiden", ax=axes[0, 1], show=False)
            axes[0, 1].set_title("UMAP by Leiden Clusters")
            
            # Bottom-left: UMAP by Speed
            bp.pl.umap(bdata, color="speed", ax=axes[1, 0], show=False)
            axes[1, 0].set_title("UMAP by Speed")
            
            # Bottom-right: Speed Time Series
            bp.pl.time_series(bdata, key="speed", max_points=5000, ax=axes[1, 1], show=False)
            axes[1, 1].set_title("Speed Over Time (Downsampled)")
            
            plt.tight_layout()
            plot_path = processed_dir / f"{fname}_full_pipeline.png"
            plt.savefig(plot_path)
            plt.close()
            res["plot_saved"] = "yes"
            res["status"] = "success"
            
        except Exception as e:
            res["error"] = str(e)
            with open(error_log, "a") as ef:
                ef.write(f"[{datetime.now()}] TIER 2 FAILED for {fname}:\n")
                ef.write(traceback.format_exc() + "\n")
        
        res["total_time_sec"] = time.time() - start_total
        tier2_results.append(res)

    t2_df = pd.DataFrame(tier2_results)
    t2_df.to_csv(tier2_csv, index=False)

    # Summary Visualization
    log_message("Generating summary visualization...", log_file)
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Time vs Frames
    ax1 = fig.add_subplot(gs[0, 0])
    sns.regplot(data=t1_df[t1_df["status"]=="success"], x="n_frames", y="processing_time_sec", ax=ax1)
    ax1.set_title("Tier 1: Processing Time vs Frame Count")
    
    # 2. Likelihood Distributions
    ax2 = fig.add_subplot(gs[0, 1])
    qual_cols = ["mean_likelihood", "median_likelihood", "min_likelihood"]
    sns.boxplot(data=t1_df[t1_df["status"]=="success"][qual_cols], ax=ax2)
    ax2.axhline(0.9, color='r', linestyle='--', label='0.9 threshold')
    ax2.axhline(0.8, color='orange', linestyle='--', label='0.8 threshold')
    ax2.set_title("Tier 1: Data Quality (Likelihood)")
    ax2.legend()
    
    # 3. Frames Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(t1_df[t1_df["status"]=="success"]["n_frames"], bins=15, ax=ax3, kde=True)
    ax3.set_title("Tier 1: Distribution of Frame Counts")
    
    # 4. Success Rates
    ax4 = fig.add_subplot(gs[1, 1])
    rates = pd.DataFrame({
        "Tier": ["Tier 1", "Tier 1", "Tier 2", "Tier 2"],
        "Status": ["Success", "Fail", "Success", "Fail"],
        "Count": [
            len(t1_df[t1_df["status"]=="success"]), len(t1_df[t1_df["status"]=="fail"]),
            len(t2_df[t2_df["status"]=="success"]), len(t2_df[t2_df["status"]=="fail"])
        ]
    })
    sns.barplot(data=rates, x="Tier", y="Count", hue="Status", ax=ax4)
    ax4.set_title("Validation Success Rates")
    
    # 5. Timing Breakdown (Tier 2)
    ax5 = fig.add_subplot(gs[2, :])
    if not t2_df[t2_df["status"]=="success"].empty:
        time_cols = ["load_time", "preprocessing_time", "pca_time", "neighbors_time", "umap_time", "leiden_time"]
        avg_times = t2_df[t2_df["status"]=="success"][time_cols].mean()
        avg_times.plot(kind='bar', ax=ax5, color='skyblue')
        ax5.set_title("Tier 2: Average Time per Pipeline Step (seconds)")
        plt.xticks(rotation=45)
    
    plt.suptitle("PyRAT Dataset Validation Results", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    summary_plot = processed_dir / "summary_visualization.png"
    plt.savefig(summary_plot, dpi=150)
    plt.close()

    # Final Summary Report
    total_t1 = len(t1_df)
    fail_t1 = len(t1_df[t1_df["status"]=="fail"])
    success_t1_count = total_t1 - fail_t1
    
    total_t2 = len(t2_df)
    fail_t2 = len(t2_df[t2_df["status"]=="fail"])
    success_t2_count = total_t2 - fail_t2

    report = f"""
==================================================
PYRAT VALIDATION SUMMARY REPORT
==================================================
TIER 1 RESULTS:
- Total Files: {total_t1}
- Success: {success_t1_count} ({success_t1_count/total_t1*100:.1f}%)
- Failed: {fail_t1} ({fail_t1/total_t1*100:.1f}%)
- Avg Time per File: {t1_df["processing_time_sec"].mean():.2f}s
- Avg Time per 1k Frames: {(t1_df["processing_time_sec"] / t1_df["n_frames"] * 1000).mean():.2f}s
- Data Quality: Avg Likelihood {t1_df["mean_likelihood"].mean():.3f}
- Failed Files: {", ".join(t1_df[t1_df["status"]=="fail"]["filename"].tolist()) if fail_t1 > 0 else "None"}

TIER 2 RESULTS:
- Total Files Tested: {total_t2}
- Success: {success_t2_count} ({success_t2_count/total_t2*100:.1f}%)
- Failed: {fail_t2}
- Avg Clusters Found: {t2_df[t2_df["status"]=="success"]["n_clusters"].mean():.1f}
- Avg Total Time: {t2_df[t2_df["status"]=="success"]["total_time_sec"].mean():.2f}s
- Failed Files: {", ".join(t2_df[t2_df["status"]=="fail"]["filename"].tolist()) if fail_t2 > 0 else "None"}

OUTPUT FILES:
- Logs: {log_file}
- Errors: {error_log}
- Tier 1 CSV: {tier1_csv}
- Tier 2 CSV: {tier2_csv}
- Summary Plot: {summary_plot}
- Individual Plots: {processed_dir}/*.png

==================================================
"""
    log_message(report, log_file)
    print(report)

if __name__ == "__main__":
    main()
