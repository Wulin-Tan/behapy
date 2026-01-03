import numpy as np
import pandas as pd

from .._core._behapydata import BehapyData


def synthetic_data(n_frames: int = 1000, n_bodyparts: int = 5) -> BehapyData:
    """
    Generate synthetic behavioral data.
    """
    # Random coordinates
    X = np.random.rand(n_frames, n_bodyparts * 2).astype("float32")

    # Bodyparts
    bodyparts = [f"bp{i}" for i in range(n_bodyparts)]

    # obs
    obs = pd.DataFrame(index=[f"frame_{i}" for i in range(n_frames)])
    obs["frame_number"] = np.arange(n_frames)

    # var
    var_list = []
    for bp in bodyparts:
        var_list.append({"bodypart": bp, "coord_axis": "x"})
        var_list.append({"bodypart": bp, "coord_axis": "y"})
    var = pd.DataFrame(var_list)
    var.index = [f"{d['bodypart']}_{d['coord_axis']}" for d in var_list]

    # layers
    # Repeat likelihood for each coordinate axis (x, y) to match var dimension
    likelihood = np.random.rand(n_frames, n_bodyparts).astype("float32")
    likelihood_expanded = np.repeat(likelihood, 2, axis=1)

    layers = {"likelihood": likelihood_expanded}

    # uns
    uns = {"bodyparts": bodyparts, "software": "synthetic"}

    return BehapyData(X=X, obs=obs, var=var, layers=layers, uns=uns)
