import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import behapy as bp

def main():
    print("=== Testing Neighbors Performance: Exact vs Annoy (v0.2.3) ===")
    
    # 1. Load data
    data_path = "data/raw/dlc/pyrat/R1D1.csv"
    print(f"Loading {data_path} (36k frames)...")
    bdata_orig = bp.io.read(data_path, software='deeplabcut')
    
    # Preprocessing
    bp.pp.calculate_qc_metrics(bdata_orig)
    bp.pp.interpolate_missing(bdata_orig)
    bp.pp.smooth(bdata_orig)
    # Ensure speed_centroid exists for plotting
    from behapy.preprocessing._features import compute_features
    compute_features(bdata_orig)
    bp.tl.pca(bdata_orig, n_comps=30)
    
    # 2. Benchmark Exact Neighbors
    print("\n[Benchmarking Exact Neighbors]")
    bdata_exact = bdata_orig.copy()
    start_time = time.time()
    bp.pp.neighbors(bdata_exact, method='exact', use_rep='X_pca')
    exact_time = time.time() - start_time
    print(f"Exact Neighbors Time: {exact_time:.2f}s")

    # 2b. Benchmark Default Neighbors (Scanpy/UMAP)
    print("\n[Benchmarking Default Neighbors (Scanpy/UMAP)]")
    bdata_default = bdata_orig.copy()
    start_time = time.time()
    bp.pp.neighbors(bdata_default, method='umap', use_rep='X_pca')
    default_time = time.time() - start_time
    print(f"Default Neighbors Time: {default_time:.2f}s")
    
    # 3. Benchmark Annoy Neighbors
    print("\n[Benchmarking Annoy Neighbors]")
    bdata_annoy = bdata_orig.copy()
    start_time = time.time()
    bp.pp.neighbors(bdata_annoy, method='annoy', use_rep='X_pca', n_trees=20)
    annoy_time = time.time() - start_time
    print(f"Annoy Neighbors Time: {annoy_time:.2f}s")
    
    speedup = exact_time / annoy_time
    print(f"\nSpeedup Factor: {speedup:.2f}x")
    
    # 4. Accuracy Comparison
    print("\n[Computing Neighbor Accuracy]")
    # Extract indices for both
    def get_indices(bdata):
        dist = bdata.obsp["distances"]
        # Convert sparse to dense indices for comparison
        # Each row has n_neighbors non-zero entries
        indices = []
        n_obs = bdata.n_obs
        for i in range(n_obs):
            row = dist.getrow(i)
            indices.append(set(row.indices))
        return indices

    exact_indices = get_indices(bdata_exact)
    annoy_indices = get_indices(bdata_annoy)
    
    overlaps = []
    for i in range(len(exact_indices)):
        intersect = exact_indices[i].intersection(annoy_indices[i])
        overlaps.append(len(intersect) / 15) # n_neighbors = 15
        
    mean_accuracy = np.mean(overlaps) * 100
    print(f"Mean Neighbor Accuracy (Overlap): {mean_accuracy:.2f}%")
    
    # 5. UMAP Visual Comparison
    print("\n[Generating UMAP comparison]")
    start_umap = time.time()
    bp.tl.umap(bdata_exact)
    exact_umap_time = time.time() - start_umap
    
    start_umap = time.time()
    bp.tl.umap(bdata_annoy)
    annoy_umap_time = time.time() - start_umap
    
    print(f"Exact UMAP Time: {exact_umap_time:.2f}s")
    print(f"Annoy UMAP Time: {annoy_umap_time:.2f}s")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    bp.pl.embedding(bdata_exact, basis='umap', color='speed', ax=axes[0], show=False)
    axes[0].set_title(f"Exact Neighbors (Time: {exact_time:.2f}s)")
    
    bp.pl.embedding(bdata_annoy, basis='umap', color='speed', ax=axes[1], show=False)
    axes[1].set_title(f"Annoy Neighbors (Time: {annoy_time:.2f}s, Acc: {mean_accuracy:.1f}%)")
    
    plt.tight_layout()
    output_path = "data/processed/neighbors_performance_comparison.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nComparison plot saved to {output_path}")

if __name__ == "__main__":
    main()
