import unittest

from behapy.datasets import synthetic_data
from behapy.preprocessing import (
    center_coordinates,
    compute_acceleration,
    compute_angular_velocity,
    compute_angles,
    compute_jerk,
    compute_speed,
    egocentric_alignment,
    pixel_to_real,
)
from behapy.tools import hdbscan, hierarchical_clustering, tsne


class TestExtended(unittest.TestCase):
    def setUp(self):
        self.bdata = synthetic_data(n_frames=100, n_bodyparts=3)

    def test_speed_accel_jerk(self):
        compute_speed(self.bdata, fps=30)
        self.assertIn("speed", self.bdata.obsm)
        compute_acceleration(self.bdata, fps=30)
        self.assertIn("acceleration", self.bdata.obsm)
        compute_jerk(self.bdata, fps=30)
        self.assertIn("jerk", self.bdata.obsm)

    def test_angles_angular_vel(self):
        # Define a joint: bodypart0-bodypart1-bodypart2
        joint_dict = {"joint1": ["bp0", "bp1", "bp2"]}
        compute_angles(self.bdata, joint_dict)
        self.assertIn("angles", self.bdata.obsm)
        compute_angular_velocity(self.bdata, joint_dict)
        self.assertIn("angular_velocity", self.bdata.obsm)

    def test_transformations(self):
        # Egocentric alignment
        egocentric_alignment(
            self.bdata,
            ref_bodypart="bp0",
            heading_bodypart="bp1",
        )
        self.assertIn("egocentric", self.bdata.layers)

        # Center coordinates
        center_coordinates(self.bdata, center="mean")
        self.assertIn("center_coords", self.bdata.uns)

        # Pixel to real
        pixel_to_real(self.bdata, scale_factor=0.5)
        self.assertEqual(self.bdata.uns["scale_factor"], 0.5)

    def test_tools_extended(self):
        from behapy.tools import pca

        pca(self.bdata, n_comps=2)

        # t-SNE
        tsne(self.bdata, n_components=2)
        self.assertIn("X_tsne", self.bdata.obsm)

        # HDBSCAN
        hdbscan(self.bdata, min_cluster_size=5)
        self.assertIn("hdbscan", self.bdata.obs.columns)

        # Hierarchical
        hierarchical_clustering(self.bdata, n_clusters=3)
        self.assertIn("hierarchical", self.bdata.obs.columns)


if __name__ == "__main__":
    unittest.main()
