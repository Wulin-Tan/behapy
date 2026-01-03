import unittest
import numpy as np
import pandas as pd
from behapy._core._behapydata import BehapyData
from behapy.preprocessing import (
    normalize_total, scale, log_transform, quantile_normalization,
    neighbors, egocentric_alignment, pixel_to_real, center_coordinates
)
from behapy.tools import pca, umap, tsne, leiden, louvain, hdbscan, kmeans, rank_features_groups

class TestPhase6To11(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset
        n_frames = 100
        n_bodyparts = 5
        coords = np.random.rand(n_frames, n_bodyparts * 2).astype(np.float32)
        
        bodyparts = [f"bp{i}" for i in range(n_bodyparts)]
        var = pd.DataFrame({
            "bodypart": np.repeat(bodyparts, 2),
            "coord": ["x", "y"] * n_bodyparts
        }, index=[f"{bp}_{c}" for bp in bodyparts for c in ["x", "y"]])
        
        obs = pd.DataFrame({
            "frame": np.arange(n_frames),
            "behavior": np.random.choice(["walk", "run", "still"], n_frames)
        }, index=[f"f{i}" for i in range(n_frames)])
        
        self.bdata = BehapyData(X=coords, obs=obs, var=var)
        self.bdata.uns["bodyparts"] = bodyparts

    def test_normalization(self):
        # Test quantile normalization
        quantile_normalization(self.bdata)
        self.assertIn("quantile_normalized", self.bdata.layers)
        
        # Test normalize total
        normalize_total(self.bdata, target_sum=100)
        self.assertIn("normalized", self.bdata.layers)
        
    def test_neighbors_and_clustering(self):
        # Need PCA for most clustering/neighbors
        pca(self.bdata, n_comps=5)
        self.assertIn("X_pca", self.bdata.obsm)
        
        neighbors(self.bdata, n_neighbors=5, use_rep="X_pca")
        self.assertIn("connectivities", self.bdata.obsp)
        
        # Test clustering
        leiden(self.bdata)
        self.assertIn("leiden", self.bdata.obs.columns)
        
        kmeans(self.bdata, n_clusters=3)
        self.assertIn("kmeans", self.bdata.obs.columns)
        
        hdbscan(self.bdata, min_cluster_size=5)
        self.assertIn("hdbscan", self.bdata.obs.columns)

    def test_markers(self):
        rank_features_groups(self.bdata, groupby="behavior")
        self.assertIn("rank_features_groups", self.bdata.uns)

if __name__ == "__main__":
    unittest.main()
