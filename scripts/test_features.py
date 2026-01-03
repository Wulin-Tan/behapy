import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import behapy as bp

def main():
    print("=== Testing Kinematic Feature Engineering (v0.2.2) ===")
    
    # 1. Load data
    data_path = "data/raw/dlc/pyrat/R1D1.csv"
    print(f"Loading {data_path}...")
    bdata = bp.io.read(data_path, software='deeplabcut')
    
    # 2. Preprocessing
    print("Preprocessing...")
    bp.pp.calculate_qc_metrics(bdata)
    bp.pp.detect_outliers(bdata)
    bp.pp.interpolate_missing(bdata)
    bp.pp.smooth(bdata, method='savgol', window_length=11)
    
    # 3. Compute Features
    print("Computing features...")
    
    # Compute speed first (required for acceleration)
    bp.pp.compute_speed(bdata)
    # Extract centroid speed to obs['speed'] for compute_features
    if "speed_centroid" in bdata.obsm["speed"]:
        bdata.obs["speed"] = bdata.obsm["speed"]["speed_centroid"]
    else:
        bdata.obs["speed"] = bdata.obsm["speed"].iloc[:, 0]
        
    # Compute all standard features
    bp.pp.compute_features(bdata, features='all')
    
    # Compute distance between nose and tail_base (if they exist)
    bodyparts = bdata.uns["bodyparts"]
    bp1, bp2 = None, None
    if 'nose' in bodyparts and 'tail_base' in bodyparts:
        bp1, bp2 = 'nose', 'tail_base'
    elif len(bodyparts) >= 2:
        bp1, bp2 = bodyparts[0], bodyparts[1]
        
    if bp1 and bp2:
        print(f"Computing distance between {bp1} and {bp2}...")
        bp.pp.compute_bodypart_distance(bdata, bp1, bp2)
        dist_key = f"distance_{bp1}_{bp2}"
    else:
        dist_key = None

    # 4. Print Summary Statistics
    print("\nFeature Summary Statistics:")
    features_to_check = ["speed", "acceleration", "angular_velocity"]
    if dist_key:
        features_to_check.append(dist_key)
        
    for feat in features_to_check:
        val = bdata.obs[feat]
        print(f"- {feat:18}: mean={val.mean():.4f}, std={val.std():.4f}, max={val.max():.4f}, nans={val.isna().sum()}")

    # 5. Visualization
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Choose a bodypart for trajectory
    bp_plot = "lumbar" if "lumbar" in bodyparts else bodyparts[0]
    
    # Panel 1: Trajectory colored by speed
    bp.pl.trajectory(bdata, bodypart=bp_plot, color_by="speed", ax=axes[0, 0], show=False)
    axes[0, 0].set_title(f"Trajectory ({bp_plot}) colored by Speed")
    
    # Panel 2: Trajectory colored by acceleration
    bp.pl.trajectory(bdata, bodypart=bp_plot, color_by="acceleration", ax=axes[0, 1], show=False)
    axes[0, 1].set_title(f"Trajectory ({bp_plot}) colored by Acceleration")
    
    # Panel 3: Trajectory colored by angular_velocity
    bp.pl.trajectory(bdata, bodypart=bp_plot, color_by="angular_velocity", ax=axes[1, 0], show=False)
    axes[1, 0].set_title(f"Trajectory ({bp_plot}) colored by Angular Velocity")
    
    # Panel 4: Time series of features
    time = np.arange(len(bdata))
    axes[1, 1].plot(time, bdata.obs["speed"] / bdata.obs["speed"].max(), label="Speed (norm)", alpha=0.7)
    axes[1, 1].plot(time, bdata.obs["acceleration"] / bdata.obs["acceleration"].max(), label="Accel (norm)", alpha=0.7)
    axes[1, 1].plot(time, bdata.obs["angular_velocity"] / bdata.obs["angular_velocity"].max(), label="AngVel (norm)", alpha=0.7)
    axes[1, 1].set_title("Feature Time Series (Normalized)")
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 1000)  # Zoom in to see patterns
    
    plt.tight_layout()
    output_path = "data/processed/kinematic_features_demo.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
