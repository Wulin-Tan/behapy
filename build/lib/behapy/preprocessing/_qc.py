from typing import Optional

import numpy as np
import pandas as pd

from .._core._behapydata import BehapyData


def calculate_qc_metrics(bdata: BehapyData, likelihood_threshold: float = 0.9) -> BehapyData:
    """
    Calculate quality control metrics for frames and bodyparts.

    Parameters
    ----------
    bdata
        BehapyData object.
    likelihood_threshold
        Threshold below which a detection is considered low quality.

    Returns
    -------
    Modified BehapyData object.
    """
    if "likelihood" not in bdata.layers:
        raise ValueError("Likelihood data not found in bdata.layers['likelihood']")

    likelihood = bdata.layers["likelihood"]

    # Per-frame metrics
    bdata.obs["mean_likelihood"] = np.nanmean(likelihood, axis=1)
    bdata.obs["n_low_likelihood"] = np.sum(likelihood < likelihood_threshold, axis=1)
    bdata.obs["n_missing"] = np.sum(np.isnan(likelihood), axis=1)

    # Per-bodypart metrics
    # bdata.var has bodyparts expanded by coords (x, y).
    # We'll calculate metrics for each column in likelihood layer
    # Since likelihood is repeated for x and y, we can just take every 2nd value
    # or just use the full expanded version.

    detection_rates = np.mean(likelihood > likelihood_threshold, axis=0)
    mean_likelihoods = np.mean(likelihood, axis=0)

    # Store in var
    bdata.var["detection_rate"] = detection_rates
    bdata.var["mean_likelihood"] = mean_likelihoods

    bdata.uns["qc"] = {"likelihood_threshold": likelihood_threshold}

    return bdata


def detect_outliers(
    bdata: BehapyData, method: str = "zscore", threshold: float = 3.0
) -> BehapyData:
    """
    Detect outlier coordinates for each bodypart.

    Parameters
    ----------
    bdata
        BehapyData object.
    method
        Detection method: 'zscore', 'iqr', or 'speed'.
    threshold
        Sensitivity threshold.

    Returns
    -------
    Modified BehapyData object.
    """
    X = bdata.X
    outliers = np.zeros_like(X, dtype=bool)

    if method == "zscore":
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        z_scores = np.abs((X - mean) / (std + 1e-8))
        outliers = z_scores > threshold

    elif method == "iqr":
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (X < lower_bound) | (X > upper_bound)

    elif method == "speed":
        # Simple velocity-based outlier detection
        velocity = np.zeros_like(X)
        velocity[1:] = np.abs(np.diff(X, axis=0))
        mean_v = np.nanmean(velocity, axis=0)
        std_v = np.nanstd(velocity, axis=0)
        outliers = velocity > (mean_v + threshold * std_v)

    bdata.layers["outliers"] = outliers
    bdata.uns["outlier_detection"] = {"method": method, "threshold": threshold}

    return bdata
