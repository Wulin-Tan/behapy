from typing import List

import numpy as np


def validate_coords(coords: np.ndarray):
    """
    Check for valid coordinate array (numeric, 2D/3D).

    Parameters
    ----------
    coords
        Coordinate array of shape (n_frames, n_bodyparts * n_dims)
        or (n_frames, n_bodyparts, n_dims).
    """
    if not isinstance(coords, np.ndarray):
        raise TypeError("Coordinates must be a numpy array.")
    if not np.issubdtype(coords.dtype, np.number):
        raise TypeError("Coordinates must be numeric.")
    if coords.ndim not in [2, 3]:
        raise ValueError(f"Coordinates must be 2D or 3D, got {coords.ndim}D.")


def validate_likelihood(likelihood: np.ndarray):
    """
    Check likelihood values are between 0 and 1.

    Parameters
    ----------
    likelihood
    Likelihood array of shape (n_frames, n_bodyparts).
    """
    if not isinstance(likelihood, np.ndarray):
        raise TypeError("Likelihood must be a numpy array.")
    if not (np.nanmin(likelihood) >= 0 and np.nanmax(likelihood) <= 1):
        raise ValueError("Likelihood values must be between 0 and 1.")


def validate_bodyparts(bodyparts: List[str], coords: np.ndarray):
    """
    Check bodyparts match coordinate dimensions.

    Parameters
    ----------
    bodyparts
        List of bodypart names.
    coords
        Coordinate array.
    """
    n_bodyparts = len(bodyparts)
    if coords.ndim == 3:
        if coords.shape[1] != n_bodyparts:
            raise ValueError(
                "Number of bodyparts "
                f"({n_bodyparts}) does not match coordinates shape {coords.shape}."
            )
    elif coords.ndim == 2:
        # Assuming coords are (n_frames, n_bodyparts * n_dims)
        # We need to know n_dims, usually 2 or 3.
        if coords.shape[1] % n_bodyparts != 0:
            raise ValueError(
                f"Coordinates shape {coords.shape} is not compatible with "
                f"{n_bodyparts} bodyparts."
            )


def validate_skeleton(skeleton: List[List[str]], bodyparts: List[str]):
    """
    Check skeleton references valid bodyparts.

    Parameters
    ----------
    skeleton
        List of pairs of bodypart names defining edges.
    bodyparts
        List of valid bodypart names.
    """
    for edge in skeleton:
        if len(edge) != 2:
            raise ValueError(f"Skeleton edge must be a pair of bodyparts, got {edge}.")
        for bp in edge:
            if bp not in bodyparts:
                raise ValueError(f"Bodypart '{bp}' in skeleton not found in provided bodyparts.")


def check_bdata_is_type(bdata, cls):
    """
    Type checking utility for BehapyData.
    """
    if not isinstance(bdata, cls):
        raise TypeError(f"Expected {cls.__name__}, got {type(bdata).__name__}.")
