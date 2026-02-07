from typing import Optional, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter

from .._core._behapydata import BehapyData


def smooth_savgol(
    bdata: BehapyData,
    window_length: int = 5,
    polyorder: int = 2,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> BehapyData:
    """Apply Savitzky-Golay filter."""
    data = bdata.layers[layer] if layer else bdata.X
    smoothed = savgol_filter(data, window_length, polyorder, axis=0)

    if inplace:
        bdata.layers["smoothed"] = smoothed
        bdata.X = smoothed
        bdata.uns["smooth"] = {
            "method": "savgol",
            "window_length": window_length,
            "polyorder": polyorder,
        }

    return bdata


def smooth_gaussian(
    bdata: BehapyData, sigma: float = 1.0, layer: Optional[str] = None, inplace: bool = True
) -> BehapyData:
    """Apply Gaussian smoothing."""
    data = bdata.layers[layer] if layer else bdata.X
    smoothed = gaussian_filter1d(data, sigma, axis=0)

    if inplace:
        bdata.layers["smoothed"] = smoothed
        bdata.X = smoothed
        bdata.uns["smooth"] = {"method": "gaussian", "sigma": sigma}

    return bdata


def smooth_median(
    bdata: BehapyData, window_length: int = 5, layer: Optional[str] = None, inplace: bool = True
) -> BehapyData:
    """Apply median filter."""
    data = bdata.layers[layer] if layer else bdata.X
    # medfilt expects 1D or 2D. data is 2D (n_frames, n_features)
    # Apply along time axis for each feature
    smoothed = np.zeros_like(data)
    for i in range(data.shape[1]):
        smoothed[:, i] = medfilt(data[:, i], window_length)

    if inplace:
        bdata.layers["smoothed"] = smoothed
        bdata.X = smoothed
        bdata.uns["smooth"] = {"method": "median", "window_length": window_length}

    return bdata


def smooth(bdata: BehapyData, method: str = "savgol", **kwargs) -> BehapyData:
    """Unified interface for smoothing."""
    if method == "savgol":
        return smooth_savgol(bdata, **kwargs)
    elif method == "gaussian":
        return smooth_gaussian(bdata, **kwargs)
    elif method == "median":
        return smooth_median(bdata, **kwargs)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
