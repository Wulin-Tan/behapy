import unittest

import numpy as np
import pandas as pd

from behapy._core._behapydata import BehapyData


class TestCore(unittest.TestCase):
    def setUp(self):
        n_frames = 100
        n_bodyparts = 3
        X = np.random.rand(n_frames, n_bodyparts * 2).astype("float32")
        obs = pd.DataFrame(index=[f"f{i}" for i in range(n_frames)])
        var_list = []
        for i in range(n_bodyparts):
            var_list.append({"bodypart": f"bp{i}", "coord_axis": "x"})
            var_list.append({"bodypart": f"bp{i}", "coord_axis": "y"})
        var = pd.DataFrame(var_list)
        var.index = [f"{d['bodypart']}_{d['coord_axis']}" for d in var_list]
        self.bdata = BehapyData(X=X, obs=obs, var=var)
        self.bdata.uns["bodyparts"] = [f"bp{i}" for i in range(n_bodyparts)]

    def test_behapydata_properties(self):
        self.assertEqual(self.bdata.n_frames, 100)
        self.assertEqual(self.bdata.n_features, 6)

    def test_validation(self):
        # Should not raise error on init
        pass


if __name__ == "__main__":
    unittest.main()
