import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import behapy as bp

def main():
    print("=== Testing Temporal Analysis (v0.3.0) ===")
    
    # 1. Load data
    data_path = "data/raw/dlc/pyrat/R1D1.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading {data_path}...")
    bdata = bp.io.read(data_path, software='deeplabcut')
    
    # 2. Pipeline
    print("Running preprocessing and clustering...")
    bp.pp.calculate_qc_metrics(bdata)
    bp.pp.interpolate_missing(bdata)
    bp.pp.smooth(bdata)
    bp.pp.compute_features(bdata)
    bp.tl.pca(bdata, n_comps=30)
    bp.pp.neighbors(bdata, method='annoy')
    bp.tl.umap(bdata)
    bp.tl.leiden(bdata, resolution=0.5)
    
    # 3. Temporal Analysis
    print("\n[Computing Temporal Metrics]")
    
    # Transitions
    bp.tl.compute_transitions(bdata, key='leiden', normalize=True)
    print("Transition matrix computed.")
    
    # Entropy
    bp.tl.compute_transition_entropy(bdata, key='leiden')
    entropy = bdata.uns['leiden_transition_entropy']
    print(f"Mean Transition Entropy: {np.mean(entropy):.3f}")
    
    # Bouts
    bp.tl.detect_bouts(bdata, key='leiden', min_duration=5)
    bouts = bdata.uns['leiden_bouts']
    print(f"Detected {len(bouts)} bouts (min_duration=5)")
    
    # Bout Stats
    bp.tl.compute_bout_statistics(bdata, key='leiden')
    bout_stats = bdata.uns['leiden_bout_stats']
    print("\nTop 5 clusters by total frames:")
    print(bout_stats.sort_values('total_frames', ascending=False).head())
    
    # 4. Visualization
    print("\n[Generating Visualizations]")
    fig = plt.figure(figsize=(15, 12))
    
    # Transition Matrix
    ax1 = fig.add_subplot(2, 2, 1)
    bp.pl.transition_matrix(bdata, key='leiden', ax=ax1)
    
    # Bout Duration Distribution
    ax2 = fig.add_subplot(2, 2, 2)
    # Filter to top 5 clusters for clarity
    top_clusters = bout_stats.sort_values('total_frames', ascending=False).index[:5].tolist()
    bp.pl.bout_duration_distribution(bdata, key='leiden', clusters=top_clusters, ax=ax2)
    
    # Ethogram
    ax3 = fig.add_subplot(2, 1, 2)
    bp.pl.ethogram_temporal(bdata, key='leiden', start=0, end=5000, ax=ax3)
    
    plt.tight_layout()
    output_path = "data/processed/temporal_analysis_demo.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    
    # 5. Print Summary
    print("\n=== Summary Results ===")
    trans_matrix = bdata.uns['leiden_transitions']
    labels = bdata.uns['leiden_transition_labels']
    
    # Find top transitions (excluding self-transitions)
    off_diag = trans_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    max_idx = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    print(f"Strongest off-diagonal transition: Cluster {labels[max_idx[0]]} -> {labels[max_idx[1]]} (p={off_diag[max_idx]:.3f})")

if __name__ == "__main__":
    main()
