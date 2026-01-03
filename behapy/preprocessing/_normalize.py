from typing import Optional, Union

import numpy as np

from .._core._behapydata import BehapyData


def normalize_total(
    bdata: BehapyData, target_sum: float = 1e4, layer: Optional[str] = None, inplace: bool = True
) -> Optional[BehapyData]:
    """
    Normalize each frame so feature values sum to target_sum.
    """
    if not inplace:
        bdata = bdata.copy()

    X = bdata.layers[layer] if layer else bdata.X
    counts_per_frame = np.sum(X, axis=1)
    # Avoid division by zero
    counts_per_frame[counts_per_frame == 0] = 1

    X_norm = X * (target_sum / counts_per_frame[:, np.newaxis])

    if inplace:
        bdata.layers["normalized"] = X_norm
        bdata.X = X_norm
        bdata.uns["normalize"] = {"method": "total", "target_sum": target_sum}
        return None
    else:
        bdata.layers["normalized"] = X_norm
        bdata.X = X_norm
        return bdata


def scale(
    bdata: BehapyData,
    zero_center: bool = True,
    max_value: Optional[float] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[BehapyData]:
    """
    Z-score standardization: (X - mean) / std.
    """
    if not inplace:
        bdata = bdata.copy()

    X = bdata.layers[layer] if layer else bdata.X

    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1

    X_scaled = X.copy()
    if zero_center:
        X_scaled -= mean
    X_scaled /= std

    if max_value is not None:
        X_scaled = np.clip(X_scaled, -max_value, max_value)

    if inplace:
        bdata.layers["scaled"] = X_scaled
        bdata.X = X_scaled
        bdata.uns["scale"] = {
            "mean": mean,
            "std": std,
            "zero_center": zero_center,
            "max_value": max_value,
        }
        return None
    else:
        bdata.layers["scaled"] = X_scaled
        bdata.X = X_scaled
        return bdata


def log_transform(
    bdata: BehapyData,
    base: Optional[float] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[BehapyData]:
    """
    Apply log transformation: log(X + 1).
    """
    if not inplace:
        bdata = bdata.copy()

    X = bdata.layers[layer] if layer else bdata.X

    if base is None:
        X_log = np.log1p(X)
    else:
        X_log = np.log(X + 1) / np.log(base)

    if inplace:
        bdata.layers["log"] = X_log
        bdata.X = X_log
        return None
    else:
        bdata.layers["log"] = X_log
        bdata.X = X_log
        return bdata


def quantile_normalization(
    bdata: BehapyData, layer: Optional[str] = None, inplace: bool = True
) -> Optional[BehapyData]:
    """
    Perform quantile normalization.

    Parameters
    ----------
    bdata
        BehapyData object.
    layer
        Layer to normalize. If None, uses bdata.X.
    inplace
        Whether to modify bdata in place.

    Returns
    -------
    BehapyData object if inplace=False, else None.
    """
    if not inplace:
        bdata = bdata.copy()

    X = bdata.layers[layer] if layer else bdata.X

    # Quantile normalization implementation
    X_sorted = np.sort(X, axis=0)
    X_mean = np.mean(X_sorted, axis=1)

    X_norm = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_norm[:, i] = X_mean[np.argsort(np.argsort(X[:, i]))]

    if inplace:
        bdata.layers["quantile_normalized"] = X_norm
        bdata.X = X_norm
        bdata.uns["normalize"] = {"method": "quantile"}
        return None
    else:
        bdata.layers["quantile_normalized"] = X_norm
        bdata.X = X_norm
        return bdata
