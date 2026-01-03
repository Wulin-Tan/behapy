from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
import pandas as pd

from .._core._behapydata import BehapyData


def read_sleap_h5(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read SLEAP H5 file structure and extract tracking data.

    Parameters
    ----------
    filepath
        Path to the SLEAP H5 file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'coords': numpy.ndarray of shape (n_frames, n_tracks, n_nodes, 2)
        - 'likelihood': numpy.ndarray of shape (n_frames, n_tracks, n_nodes)
        - 'bodyparts': List of node names
        - 'track_names': List of track names
    """
    with h5py.File(filepath, "r") as f:
        # SLEAP H5 structure:
        # 'tracks': (n_tracks, n_nodes, n_coords, n_frames) -> Note: SLEAP storage is often transposed
        # 'node_names': (n_nodes,)
        # 'track_names': (n_tracks,)

        tracks = f["tracks"][:]
        # SLEAP tracks are (n_frames, n_nodes, 2, n_tracks) in some versions,
        # but the common export is (n_tracks, 2, n_nodes, n_frames)
        # Let's check dimensions and reorder to (n_frames, n_tracks, n_nodes, 2)

        if tracks.ndim == 4:
            # Assume (n_tracks, 2, n_nodes, n_frames)
            tracks = np.transpose(tracks, (3, 0, 2, 1))

        node_names = [n.decode() if isinstance(n, bytes) else n for n in f["node_names"][:]]
        track_names = [n.decode() if isinstance(n, bytes) else n for n in f["track_names"][:]]

        # SLEAP doesn't always have a likelihood score in the H5 export in the same way DLC does.
        # Often, missing data is NaN.
        likelihood = np.ones((tracks.shape[0], tracks.shape[1], tracks.shape[2]))
        likelihood[np.isnan(tracks[:, :, :, 0])] = 0

    return {
        "coords": tracks,
        "likelihood": likelihood,
        "bodyparts": node_names,
        "track_names": track_names,
    }


def read_sleap(filepath: Union[str, Path], animal: Optional[str] = None, **kwargs) -> BehapyData:
    """
    Read SLEAP output and return a BehapyData object.

    Parameters
    ----------
    filepath
        Path to the SLEAP H5 file.
    animal
        Animal name (track name) to extract.
    **kwargs
        Additional arguments.

    Returns
    -------
    BehapyData object.
    """
    filepath = Path(filepath)
    data = read_sleap_h5(filepath)

    tracks = data["coords"]
    track_names = data["track_names"]

    if animal is None:
        animal_idx = 0
        animal = track_names[0]
    else:
        if animal in track_names:
            animal_idx = track_names.index(animal)
        else:
            try:
                animal_idx = int(animal)
                animal = track_names[animal_idx]
            except (ValueError, IndexError):
                raise ValueError(f"Animal {animal} not found in {track_names}")

    # Extract data for the specified animal
    # tracks shape: (n_frames, n_tracks, n_nodes, 2)
    animal_coords = tracks[:, animal_idx, :, :]  # (n_frames, n_nodes, 2)
    animal_likelihood = data["likelihood"][:, animal_idx, :]  # (n_frames, n_nodes)

    n_frames = animal_coords.shape[0]

    # Reshape coords for X: (n_frames, n_bodyparts * 2)
    X = animal_coords.reshape(n_frames, -1)

    # Create obs
    obs = pd.DataFrame(index=[f"frame_{i}" for i in range(n_frames)])
    obs["frame_number"] = np.arange(n_frames)
    obs["animal"] = animal

    # Create var
    var_list = []
    for bp in data["bodyparts"]:
        var_list.append({"bodypart": bp, "coord_axis": "x"})
        var_list.append({"bodypart": bp, "coord_axis": "y"})
    var = pd.DataFrame(var_list)
    var.index = [f"{d['bodypart']}_{d['coord_axis']}" for d in var_list]

    # Create layers
    # Repeat likelihood for each coordinate axis (x, y) to match var dimension
    likelihood_expanded = np.repeat(animal_likelihood, 2, axis=1)

    layers = {"raw": X.copy(), "likelihood": likelihood_expanded}

    # Create uns
    uns = {
        "software": "sleap",
        "filepath": str(filepath),
        "bodyparts": data["bodyparts"],
        "track_names": track_names,
    }

    return BehapyData(X=X, obs=obs, var=var, layers=layers, uns=uns)
