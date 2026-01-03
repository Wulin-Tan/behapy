import behapy as bp
import matplotlib.pyplot as plt
import os

def test_cluster_merging():
    print("Loading R1D1.csv for cluster merging test...")
    data_path = "PyRAT_dataset/R1D1.csv"
    
    # 1. Full pipeline
    bdata = bp.io.read(data_path, software='deeplabcut')
    bp.pp.calculate_qc_metrics(bdata)
    bp.pp.smooth(bdata)
    bp.pp.compute_speed(bdata)
    bp.tl.pca(bdata)
    bp.pp.neighbors(bdata)
    bp.tl.umap(bdata)
    bp.tl.leiden(bdata, resolution=1.0) # Fine-grained
    
    n_orig = len(bdata.obs['leiden'].unique())
    print(f"Original clustering: {n_orig} clusters")
    
    # 2. merge_clusters with method='hierarchy'
    print("Merging clusters (hierarchy, n=10)...")
    bp.tl.merge_clusters(bdata, key='leiden', method='hierarchy', n_clusters=10)
    
    # 3. coarse_grain_clusters with target_n=8
    print("Coarse-graining clusters (target_n=8)...")
    bp.tl.coarse_grain_clusters(bdata, key='leiden', target_n=8)
    
    # 4. Visualization
    print("Generating comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    bp.pl.umap(bdata, color='leiden', ax=axes[0], show=False)
    axes[0].set_title(f"Original ({n_orig} clusters)")
    
    bp.pl.umap(bdata, color='leiden_merged', ax=axes[1], show=False)
    axes[1].set_title("Merged (Hierarchy, n=10)")
    
    bp.pl.umap(bdata, color='leiden_coarse', ax=axes[2], show=False)
    axes[2].set_title("Coarse-grained (Target n=8)")
    
    plt.tight_layout()
    output_path = "data/processed/cluster_merging_comparison.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    test_cluster_merging()
