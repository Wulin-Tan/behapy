from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._core._behapydata import BehapyData
from ._features_2d import (
    compute_acceleration_numba,
    compute_angular_velocity_numba,
    compute_angles_numba,
    compute_distances_numba,
    compute_jerk_numba,
    compute_speed_numba,
    compute_velocity_numba,
)


def compute_distances(
    bdata: BehapyData,
    bodypart_pairs: Optional[List[Tuple[str, str]]] = None,
    store_as: str = "distances",
) -> BehapyData:
    """
    Compute Euclidean distances between pairs of bodyparts.

    Parameters
    ----------
    bdata
        BehapyData object.
    bodypart_pairs
        List of tuples containing pairs of bodyparts. If None, all pairs are computed.
    store_as
        Key to store distances in bdata.obsm.

    Returns
    -------
    BehapyData object with distances in bdata.obsm.
    """
    bodyparts = bdata.uns["bodyparts"]
    if bodypart_pairs is None:
        from itertools import combinations

        bodypart_pairs = list(combinations(bodyparts, 2))

    # Prepare indices for Numba
    pairs_idx = []
    bp_to_idx = {bp: i for i, bp in enumerate(bodyparts)}
    for bp1, bp2 in bodypart_pairs:
        pairs_idx.append([bp_to_idx[bp1], bp_to_idx[bp2]])
    pairs_idx = np.array(pairs_idx, dtype=np.int32)

    # Reshape X to (n_frames, n_bodyparts, 2)
    coords = bdata.X.reshape(bdata.n_obs, -1, 2).astype(np.float32)

    # Compute distances
    dist_array = compute_distances_numba(coords, pairs_idx)

    # Store in obsm
    cols = [f"dist_{bp1}_{bp2}" for bp1, bp2 in bodypart_pairs]
    bdata.obsm[store_as] = pd.DataFrame(dist_array, index=bdata.obs_names, columns=cols)
    bdata.uns[f"{store_as}_pairs"] = bodypart_pairs
    return bdata


def compute_speed(
    bdata: BehapyData, bodyparts: Optional[List[str]] = None, fps: int = 30, store_as: str = "speed"
) -> BehapyData:
    """
    Compute frame-to-frame speed for bodyparts.

    Parameters
    ----------
    bdata
        BehapyData object.
    bodyparts
        List of bodyparts to compute speed for. If None, all bodyparts are used.
    fps
        Frames per second.
    store_as
        Key to store speed in bdata.obsm.

    Returns
    -------
    BehapyData object with speed in bdata.obsm.
    """
    all_bodyparts = bdata.uns["bodyparts"]
    if bodyparts is None:
        bodyparts = all_bodyparts

    # Reshape X to (n_frames, n_bodyparts, 2)
    coords = bdata.X.reshape(bdata.n_obs, -1, 2).astype(np.float32)

    # Compute velocity and speed
    velocity = compute_velocity_numba(coords, float(fps))
    speed_array = compute_speed_numba(velocity)

    # Select requested bodyparts
    bp_to_idx = {bp: i for i, bp in enumerate(all_bodyparts)}
    indices = [bp_to_idx[bp] for bp in bodyparts]
    speed_array = speed_array[:, indices]

    # Store in obsm
    cols = [f"speed_{bp}" for bp in bodyparts]
    bdata.obsm[store_as] = pd.DataFrame(speed_array, index=bdata.obs_names, columns=cols)
    bdata.uns["fps"] = fps
    return bdata


def _compute_acceleration_obsm(
    bdata: BehapyData,
    bodyparts: Optional[List[str]] = None,
    fps: int = 30,
    store_as: str = "acceleration",
) -> BehapyData:
    """
    Compute frame-to-frame acceleration for bodyparts.

    Parameters
    ----------
    bdata
        BehapyData object.
    bodyparts
        List of bodyparts to compute acceleration for. If None, all bodyparts are used.
    fps
        Frames per second.
    store_as
        Key to store acceleration in bdata.obsm.

    Returns
    -------
    BehapyData object with acceleration in bdata.obsm.
    """
    all_bodyparts = bdata.uns["bodyparts"]
    if bodyparts is None:
        bodyparts = all_bodyparts

    # Reshape X to (n_frames, n_bodyparts, 2)
    coords = bdata.X.reshape(bdata.n_obs, -1, 2).astype(np.float32)

    # Compute velocity, then acceleration, then magnitude
    velocity = compute_velocity_numba(coords, float(fps))
    accel_vec = compute_acceleration_numba(velocity, float(fps))
    accel_mag = compute_speed_numba(accel_vec)  # Reuse speed_numba for magnitude

    # Select requested bodyparts
    bp_to_idx = {bp: i for i, bp in enumerate(all_bodyparts)}
    indices = [bp_to_idx[bp] for bp in bodyparts]
    accel_mag = accel_mag[:, indices]

    # Store in obsm
    cols = [f"accel_{bp}" for bp in bodyparts]
    bdata.obsm[store_as] = pd.DataFrame(accel_mag, index=bdata.obs_names, columns=cols)
    return bdata


def compute_jerk(
    bdata: BehapyData,
    bodyparts: Optional[List[str]] = None,
    fps: int = 30,
    store_as: str = "jerk",
) -> BehapyData:
    """
    Compute frame-to-frame jerk for bodyparts.

    Parameters
    ----------
    bdata
        BehapyData object.
    bodyparts
        List of bodyparts to compute jerk for. If None, all bodyparts are used.
    fps
        Frames per second.
    store_as
        Key to store jerk in bdata.obsm.

    Returns
    -------
    BehapyData object with jerk in bdata.obsm.
    """
    all_bodyparts = bdata.uns["bodyparts"]
    if bodyparts is None:
        bodyparts = all_bodyparts

    # Reshape X to (n_frames, n_bodyparts, 2)
    coords = bdata.X.reshape(bdata.n_obs, -1, 2).astype(np.float32)

    # Compute derivatives
    velocity = compute_velocity_numba(coords, float(fps))
    accel = compute_acceleration_numba(velocity, float(fps))
    jerk_vec = compute_jerk_numba(accel, float(fps))
    jerk_mag = compute_speed_numba(jerk_vec)

    # Select requested bodyparts
    bp_to_idx = {bp: i for i, bp in enumerate(all_bodyparts)}
    indices = [bp_to_idx[bp] for bp in bodyparts]
    jerk_mag = jerk_mag[:, indices]

    # Store in obsm
    cols = [f"jerk_{bp}" for bp in bodyparts]
    bdata.obsm[store_as] = pd.DataFrame(jerk_mag, index=bdata.obs_names, columns=cols)
    return bdata


def compute_angles(
    bdata: BehapyData, joint_dict: Dict[str, List[str]], store_as: str = "angles"
) -> BehapyData:
    """
    Compute angles for defined joints.

    Parameters
    ----------
    bdata
        BehapyData object.
    joint_dict
        Dictionary mapping joint names to lists of 3 bodyparts [p1, p2, p3].
        Angle is computed at p2.
    store_as
        Key to store angles in bdata.obsm.

    Returns
    -------
    BehapyData object with angles in bdata.obsm.
    """
    bodyparts = bdata.uns["bodyparts"]
    bp_to_idx = {bp: i for i, bp in enumerate(bodyparts)}

    # Prepare indices for Numba
    joints_idx = []
    joint_names = []
    for name, (p1, p2, p3) in joint_dict.items():
        joints_idx.append([bp_to_idx[p1], bp_to_idx[p2], bp_to_idx[p3]])
        joint_names.append(name)
    joints_idx = np.array(joints_idx, dtype=np.int32)

    # Reshape X to (n_frames, n_bodyparts, 2)
    coords = bdata.X.reshape(bdata.n_obs, -1, 2).astype(np.float32)

    # Compute angles
    angle_array = compute_angles_numba(coords, joints_idx)

    # Store in obsm
    bdata.obsm[store_as] = pd.DataFrame(angle_array, index=bdata.obs_names, columns=joint_names)
    bdata.uns[f"{store_as}_joints"] = joint_dict
    return bdata


def _compute_angular_velocity_obsm(
    bdata: BehapyData,
    joint_dict: Optional[Dict[str, List[str]]] = None,
    fps: int = 30,
    store_as: str = "angular_velocity",
) -> BehapyData:
    """
    Compute angular velocity for defined joints.

    Parameters
    ----------
    bdata
        BehapyData object.
    joint_dict
        Dictionary mapping joint names to lists of 3 bodyparts.
        If None, uses joints from 'angles' in bdata.obsm.
    fps
        Frames per second.
    store_as
        Key to store angular velocity in bdata.obsm.

    Returns
    -------
    BehapyData object with angular velocity in bdata.obsm.
    """
    if joint_dict is None:
        if "angles_joints" in bdata.uns:
            joint_dict = bdata.uns["angles_joints"]
        else:
            raise ValueError(
                "joint_dict must be provided or compute_angles must be run first."
            )

    # Ensure angles are computed
    if "angles" not in bdata.obsm:
        compute_angles(bdata, joint_dict)

    angles = bdata.obsm["angles"].values.astype(np.float32)

    # Compute angular velocity
    angular_vel_array = compute_angular_velocity_numba(angles, float(fps))

    # Store in obsm
    bdata.obsm[store_as] = pd.DataFrame(
        angular_vel_array, index=bdata.obs_names, columns=list(joint_dict.keys())
    )
    return bdata


def compute_acceleration(
    bdata: BehapyData, key: str = "speed", copy: bool = False
) -> Optional[BehapyData]:
    """
    Calculate frame-to-frame change in speed.

    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing speed values.
    copy
        Whether to return a copy or modify in place.

    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()

    if key not in bdata.obs:
        raise ValueError(f"Key '{key}' not found in bdata.obs. Run compute_speed first.")

    speed = bdata.obs[key].values
    accel = np.zeros_like(speed)

    # Central difference for internal points, forward/backward for edges
    accel[1:-1] = (speed[2:] - speed[:-2]) / 2.0
    accel[0] = speed[1] - speed[0]
    accel[-1] = speed[-1] - speed[-2]

    bdata.obs["acceleration"] = accel

    return bdata if copy else None


def compute_angular_velocity(
    bdata: BehapyData, bodypart: str = "centroid", copy: bool = False
) -> Optional[BehapyData]:
    """
    Calculate turning rate (change in heading direction).

    Parameters
    ----------
    bdata
        BehapyData object.
    bodypart
        Bodypart to use for heading calculation.
    copy
        Whether to return a copy or modify in place.

    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()

    # Get coordinates for requested bodypart
    bodyparts = bdata.uns["bodyparts"]
    if bodypart not in bodyparts:
        if "centroid" in bodyparts:
            bodypart = "centroid"
        else:
            bodypart = bodyparts[0]

    bp_idx = bodyparts.index(bodypart)
    coords = bdata.X.reshape(bdata.n_obs, -1, 2)[:, bp_idx, :]

    # Calculate heading direction (angle of velocity vector)
    dx = np.diff(coords[:, 0], prepend=coords[0, 0])
    dy = np.diff(coords[:, 1], prepend=coords[0, 1])
    heading = np.arctan2(dy, dx)

    # Calculate change in heading (angular velocity)
    # Handle angle wrapping
    ang_vel = np.diff(heading, prepend=heading[0])
    ang_vel = (ang_vel + np.pi) % (2 * np.pi) - np.pi

    bdata.obs["angular_velocity"] = ang_vel

    return bdata if copy else None


def compute_bodypart_distance(
    bdata: BehapyData, bp1: str, bp2: str, copy: bool = False
) -> Optional[BehapyData]:
    """
    Calculate Euclidean distance between two bodyparts.

    Parameters
    ----------
    bdata
        BehapyData object.
    bp1, bp2
        Bodypart names.
    copy
        Whether to return a copy or modify in place.

    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()

    bodyparts = bdata.uns["bodyparts"]
    if bp1 not in bodyparts or bp2 not in bodyparts:
        raise ValueError(f"Bodyparts {bp1} or {bp2} not found in bdata.")

    idx1 = bodyparts.index(bp1)
    idx2 = bodyparts.index(bp2)

    coords = bdata.X.reshape(bdata.n_obs, -1, 2)
    p1 = coords[:, idx1, :]
    p2 = coords[:, idx2, :]

    dist = np.sqrt(np.sum((p1 - p2) ** 2, axis=1))
    bdata.obs[f"distance_{bp1}_{bp2}"] = dist

    return bdata if copy else None


def compute_bodypart_angle(
    bdata: BehapyData, bp1: str, bp2: str, bp3: str, copy: bool = False
) -> Optional[BehapyData]:
    """
    Calculate angle formed by three bodyparts (bp2 is vertex).

    Parameters
    ----------
    bdata
        BehapyData object.
    bp1, bp2, bp3
        Bodypart names. bp2 is the vertex.
    copy
        Whether to return a copy or modify in place.

    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()

    bodyparts = bdata.uns["bodyparts"]
    for bp in [bp1, bp2, bp3]:
        if bp not in bodyparts:
            raise ValueError(f"Bodypart {bp} not found in bdata.")

    idx1 = bodyparts.index(bp1)
    idx2 = bodyparts.index(bp2)
    idx3 = bodyparts.index(bp3)

    coords = bdata.X.reshape(bdata.n_obs, -1, 2)
    p1 = coords[:, idx1, :]
    p2 = coords[:, idx2, :]
    p3 = coords[:, idx3, :]

    # Vectors
    v1 = p1 - p2
    v2 = p3 - p2

    # Norms
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)

    # Dot product
    dot = np.sum(v1 * v2, axis=1)

    # Cosine
    cos_theta = dot / (n1 * n2 + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)
    bdata.obs[f"angle_{bp1}_{bp2}_{bp3}"] = angle

    return bdata if copy else None


def compute_features(
    bdata: BehapyData, features: Union[str, List[str]] = "all", copy: bool = False
) -> Optional[BehapyData]:
    """
    Unified interface to compute multiple kinematic features.

    Parameters
    ----------
    bdata
        BehapyData object.
    features
        Features to compute. 'all' or list of feature names.
    copy
        Whether to return a copy or modify in place.

    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()

    if features == "all":
        features = ["speed", "acceleration", "angular_velocity"]

    if "speed" in features:
        # Check if speed exists, if not, compute it (default centroid)
        if "speed" not in bdata.obs:
            from ._features import compute_speed

            # compute_speed currently stores in obsm, we want it in obs for this API
            # Let's adapt it or call it and extract
            compute_speed(bdata)
            # Use speed of the first bodypart or centroid as default 'speed' in obs
            if "speed_centroid" in bdata.obsm["speed"]:
                bdata.obs["speed"] = bdata.obsm["speed"]["speed_centroid"]
            else:
                bdata.obs["speed"] = bdata.obsm["speed"].iloc[:, 0]

    if "acceleration" in features:
        compute_acceleration(bdata, key="speed")

    if "angular_velocity" in features:
        compute_angular_velocity(bdata)

    return bdata if copy else None
