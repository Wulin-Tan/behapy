import unittest

from behapy.datasets import synthetic_data
from behapy.preprocessing import neighbors
from behapy.tools import leiden, pca, umap


class TestTools(unittest.TestCase):
    def setUp(self):
        self.bdata = synthetic_data(n_frames=100, n_bodyparts=3)

    def test_pca(self):
        pca(self.bdata, n_comps=2)
        self.assertIn("X_pca", self.bdata.obsm)
        self.assertEqual(self.bdata.obsm["X_pca"].shape[1], 2)

    def test_neighbors_umap_leiden(self):
        pca(self.bdata, n_comps=2)
        neighbors(self.bdata, n_neighbors=5, use_rep="X_pca")
        self.assertIn("connectivities", self.bdata.obsp)

        umap(self.bdata)
        self.assertIn("X_umap", self.bdata.obsm)

        leiden(self.bdata)
        self.assertIn("leiden", self.bdata.obs.columns)


if __name__ == "__main__":
    unittest.main()
