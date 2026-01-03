import unittest

import numpy as np

from behapy.datasets import synthetic_data
from behapy.preprocessing import (
    calculate_qc_metrics,
    compute_distances,
    filter_frames,
    scale,
    smooth_savgol,
)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.bdata = synthetic_data(n_frames=100, n_bodyparts=3)

    def test_qc_metrics(self):
        calculate_qc_metrics(self.bdata)
        self.assertIn("mean_likelihood", self.bdata.obs.columns)
        self.assertIn("qc", self.bdata.uns)

    def test_filtering(self):
        calculate_qc_metrics(self.bdata)
        bdata_filt = filter_frames(
            self.bdata,
            min_likelihood=0.1,
            inplace=False,
        )
        self.assertLessEqual(bdata_filt.n_frames, self.bdata.n_frames)

    def test_smoothing(self):
        X_orig = self.bdata.X.copy()
        smooth_savgol(self.bdata, window_length=5, polyorder=2)
        self.assertFalse(np.array_equal(self.bdata.X, X_orig))

    def test_features(self):
        compute_distances(self.bdata)
        self.assertIn("distances", self.bdata.obsm)

    def test_scale(self):
        scale(self.bdata)
        self.assertIn("scaled", self.bdata.layers)


if __name__ == "__main__":
    unittest.main()
