from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_distances_numba(coords: np.ndarray, pairs_idx: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated distance calculation.

    Parameters
    ----------
    coords
        Array of shape (n_frames, n_bodyparts, 2)
    pairs_idx
        Array of shape (n_pairs, 2) containing indices of bodyparts.

    Returns
    -------
    Distances of shape (n_frames, n_pairs)
    """
    n_frames = coords.shape[0]
    n_pairs = pairs_idx.shape[0]
    distances = np.zeros((n_frames, n_pairs), dtype=np.float32)

    for j in prange(n_pairs):
        idx1 = pairs_idx[j, 0]
        idx2 = pairs_idx[j, 1]
        for i in range(n_frames):
            dx = coords[i, idx1, 0] - coords[i, idx2, 0]
            dy = coords[i, idx1, 1] - coords[i, idx2, 1]
            distances[i, j] = np.sqrt(dx * dx + dy * dy)

    return distances


@njit(parallel=True)
def compute_angles_numba(coords: np.ndarray, joints_idx: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated angle calculation.

    Parameters
    ----------
    coords
        Array of shape (n_frames, n_bodyparts, 2)
    joints_idx
        Array of shape (n_joints, 3) containing indices of bodyparts (p1, p2, p3).
        Angle is at p2.

    Returns
    -------
    Angles in degrees of shape (n_frames, n_joints)
    """
    n_frames = coords.shape[0]
    n_joints = joints_idx.shape[0]
    angles = np.zeros((n_frames, n_joints), dtype=np.float32)

    for j in prange(n_joints):
        p1_idx = joints_idx[j, 0]
        p2_idx = joints_idx[j, 1]
        p3_idx = joints_idx[j, 2]

        for i in range(n_frames):
            v1x = coords[i, p1_idx, 0] - coords[i, p2_idx, 0]
            v1y = coords[i, p1_idx, 1] - coords[i, p2_idx, 1]
            v2x = coords[i, p3_idx, 0] - coords[i, p2_idx, 0]
            v2y = coords[i, p3_idx, 1] - coords[i, p2_idx, 1]

            dot = v1x * v2x + v1y * v2y
            v1_norm = np.sqrt(v1x * v1x + v1y * v1y)
            v2_norm = np.sqrt(v2x * v2x + v2y * v2y)

            cos_theta = dot / (v1_norm * v2_norm + 1e-8)
            if cos_theta > 1.0:
                cos_theta = 1.0
            if cos_theta < -1.0:
                cos_theta = -1.0

            angles[i, j] = np.arccos(cos_theta) * 180.0 / np.pi

    return angles


@njit(parallel=True)
def compute_velocity_numba(coords: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute velocity (dx/dt, dy/dt).

    Returns
    -------
    Velocity of shape (n_frames, n_bodyparts, 2)
    """
    n_frames, n_bodyparts, n_dims = coords.shape
    velocity = np.zeros((n_frames, n_bodyparts, n_dims), dtype=np.float32)

    for j in prange(n_bodyparts):
        for i in range(1, n_frames):
            velocity[i, j, 0] = (coords[i, j, 0] - coords[i - 1, j, 0]) * fps
            velocity[i, j, 1] = (coords[i, j, 1] - coords[i - 1, j, 1]) * fps

    return velocity


@njit(parallel=True)
def compute_speed_numba(velocity: np.ndarray) -> np.ndarray:
    """
    Compute speed from velocity.

    Returns
    -------
    Speed of shape (n_frames, n_bodyparts)
    """
    n_frames, n_bodyparts, _ = velocity.shape
    speed = np.zeros((n_frames, n_bodyparts), dtype=np.float32)

    for j in prange(n_bodyparts):
        for i in range(n_frames):
            vx = velocity[i, j, 0]
            vy = velocity[i, j, 1]
            speed[i, j] = np.sqrt(vx * vx + vy * vy)

    return speed


@njit(parallel=True)
def compute_acceleration_numba(velocity: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute acceleration from velocity.

    Returns
    -------
    Acceleration of shape (n_frames, n_bodyparts, 2)
    """
    n_frames, n_bodyparts, n_dims = velocity.shape
    accel = np.zeros((n_frames, n_bodyparts, n_dims), dtype=np.float32)

    for j in prange(n_bodyparts):
        for i in range(1, n_frames):
            accel[i, j, 0] = (velocity[i, j, 0] - velocity[i - 1, j, 0]) * fps
            accel[i, j, 1] = (velocity[i, j, 1] - velocity[i - 1, j, 1]) * fps

    return accel


@njit(parallel=True)
def compute_jerk_numba(accel: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute jerk from acceleration.

    Returns
    -------
    Jerk of shape (n_frames, n_bodyparts, 2)
    """
    n_frames, n_bodyparts, n_dims = accel.shape
    jerk = np.zeros((n_frames, n_bodyparts, n_dims), dtype=np.float32)

    for j in prange(n_bodyparts):
        for i in range(1, n_frames):
            jerk[i, j, 0] = (accel[i, j, 0] - accel[i - 1, j, 0]) * fps
            jerk[i, j, 1] = (accel[i, j, 1] - accel[i - 1, j, 1]) * fps

    return jerk


@njit(parallel=True)
def compute_angular_velocity_numba(angles: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute angular velocity from angles.

    Returns
    -------
    Angular velocity of shape (n_frames, n_joints)
    """
    n_frames, n_joints = angles.shape
    angular_vel = np.zeros((n_frames, n_joints), dtype=np.float32)

    for j in prange(n_joints):
        for i in range(1, n_frames):
            diff = angles[i, j] - angles[i - 1, j]
            # Handle angle wrap around if necessary (not needed for degrees 0-180)
            angular_vel[i, j] = diff * fps

    return angular_vel
