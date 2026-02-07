from typing import Optional, Tuple, Union

import numpy as np

from .._core._behapydata import BehapyData


def egocentric_alignment(
    bdata: BehapyData,
    ref_bodypart: str,
    heading_bodypart: Optional[str] = None,
    inplace: bool = True,
) -> Optional[BehapyData]:
    """
    Translate and rotate coordinates for egocentric alignment.
    """
    if not inplace:
        bdata = bdata.copy()

    X = bdata.X.copy()
    n_frames = X.shape[0]
    bodyparts = bdata.uns["bodyparts"]

    # Get reference bodypart indices
    ref_idx = np.where(bdata.var["bodypart"] == ref_bodypart)[0]
    ref_coords = X[:, ref_idx]  # (n_frames, 2)

    # 1. Translation: move ref_bodypart to (0,0)
    for i in range(len(bodyparts)):
        bp_idx = np.where(bdata.var["bodypart"] == bodyparts[i])[0]
        X[:, bp_idx] -= ref_coords

    # 2. Rotation: if heading_bodypart provided, rotate so ref->heading points along x-axis
    if heading_bodypart:
        head_idx = np.where(bdata.var["bodypart"] == heading_bodypart)[0]
        head_coords = X[:, head_idx]

        # Calculate angle to x-axis
        angles = np.arctan2(head_coords[:, 1], head_coords[:, 0])

        # Rotate all bodyparts
        for i in range(len(bodyparts)):
            bp_idx = np.where(bdata.var["bodypart"] == bodyparts[i])[0]
            curr_coords = X[:, bp_idx]

            c, s = np.cos(-angles), np.sin(-angles)

            x_new = curr_coords[:, 0] * c - curr_coords[:, 1] * s
            y_new = curr_coords[:, 0] * s + curr_coords[:, 1] * c

            X[:, bp_idx[0]] = x_new
            X[:, bp_idx[1]] = y_new

    if inplace:
        bdata.layers["egocentric"] = X
        bdata.X = X
        bdata.uns["egocentric"] = {"ref": ref_bodypart, "heading": heading_bodypart}
        return None
    else:
        bdata.layers["egocentric"] = X
        bdata.X = X
        return bdata


def pixel_to_real(
    bdata: BehapyData,
    scale_factor: Optional[float] = None,
    pixel_range: Optional[Tuple[float, float]] = None,
    real_range: Optional[Tuple[float, float]] = None,
    inplace: bool = True,
) -> Optional[BehapyData]:
    """
    Convert pixel coordinates to real-world units.
    """
    if scale_factor is None:
        if pixel_range and real_range:
            scale_factor = (real_range[1] - real_range[0]) / (pixel_range[1] - pixel_range[0])
        else:
            raise ValueError("Must provide scale_factor or both pixel_range and real_range")

    if not inplace:
        bdata = bdata.copy()

    bdata.X *= scale_factor
    for layer in bdata.layers:
        if layer != "likelihood" and layer != "outliers":
            bdata.layers[layer] *= scale_factor

    bdata.uns["scale_factor"] = scale_factor

    return bdata if not inplace else None


def center_coordinates(
    bdata: BehapyData, center: str = "mean", inplace: bool = True
) -> Optional[BehapyData]:
    """
    Center coordinates around 'mean' or 'median'.
    """
    if not inplace:
        bdata = bdata.copy()

    X = bdata.X
    if center == "mean":
        center_val = np.nanmean(X, axis=0)
    elif center == "median":
        center_val = np.nanmedian(X, axis=0)
    else:
        raise ValueError("center must be 'mean' or 'median'")

    bdata.X -= center_val
    for layer in bdata.layers:
        if layer != "likelihood" and layer != "outliers":
            bdata.layers[layer] -= center_val

    bdata.uns["center_coords"] = {"method": center, "center_val": center_val}

    return bdata if not inplace else None
