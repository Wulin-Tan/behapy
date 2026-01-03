from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .._core._behapydata import BehapyData


def read_dlc_h5(filepath: Union[str, Path], animal: Optional[str] = None) -> Dict[str, Any]:
    """
    Read DeepLabCut (DLC) H5 file using pandas.

    Parameters
    ----------
    filepath
        Path to the DLC H5 file.
    animal
        Animal name if the file contains multi-animal data. If None, the first animal found is used.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'coords': numpy.ndarray of shape (n_frames, n_bodyparts, 2)
        - 'likelihood': numpy.ndarray of shape (n_frames, n_bodyparts)
        - 'bodyparts': List of bodypart names
        - 'scorer': The name of the scorer/DLC model
        - 'animal': The animal name extracted
    """
    df = pd.read_hdf(filepath)
    return _process_dlc_df(df, animal)


def read_dlc_csv(filepath: Union[str, Path], animal: Optional[str] = None) -> Dict[str, Any]:
    """
    Read DeepLabCut (DLC) CSV file.

    Parameters
    ----------
    filepath
        Path to the DLC CSV file.
    animal
        Animal name if the file contains multi-animal data. If None, the first animal found is used.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'coords': numpy.ndarray of shape (n_frames, n_bodyparts, 2)
        - 'likelihood': numpy.ndarray of shape (n_frames, n_bodyparts)
        - 'bodyparts': List of bodypart names
        - 'scorer': The name of the scorer/DLC model
        - 'animal': The animal name extracted
    """
    # DLC CSVs usually have 3 or 4 header rows
    # 0: scorer, 1: bodyparts, 2: coords (x, y, likelihood)
    # If multi-animal: 0: scorer, 1: individuals, 2: bodyparts, 3: coords

    # First, peek at the header to determine if it's multi-animal
    header_df = pd.read_csv(filepath, header=None, nrows=4)
    if header_df.iloc[1, 0] == "individuals" or "individuals" in header_df.values:
        header = [0, 1, 2, 3]
    else:
        header = [0, 1, 2]

    df = pd.read_csv(filepath, header=header, index_col=0)
    return _process_dlc_df(df, animal)


def _process_dlc_df(df: pd.DataFrame, animal: Optional[str] = None) -> Dict[str, Any]:
    """Helper to process DLC DataFrame into a dictionary."""
    if df.columns.nlevels == 4:
        # Multi-animal: (scorer, individual, bodypart, coords)
        scorer = df.columns.get_level_values(0).unique()[0]
        animals = df.columns.get_level_values(1).unique().tolist()
        if animal is None:
            animal = animals[0]
        elif animal not in animals:
            raise ValueError(f"Animal {animal} not found in {animals}")

        df_animal = df.xs(animal, level=1, axis=1)
        bodyparts = df_animal.columns.get_level_values(0).unique().tolist()
    else:
        # Single animal: (scorer, bodypart, coords)
        scorer = df.columns.get_level_values(0).unique()[0]
        bodyparts = df.columns.get_level_values(1).unique().tolist()
        animal = "animal1"
        df_animal = df.xs(scorer, level=0, axis=1)

    # Extract coords and likelihood
    coords_list = []
    likelihood_list = []

    for bp in bodyparts:
        bp_df = df_animal[bp]
        coords_list.append(bp_df[["x", "y"]].values)
        likelihood_list.append(bp_df["likelihood"].values)

    coords = np.stack(coords_list, axis=1)  # (n_frames, n_bodyparts, 2)
    likelihood = np.stack(likelihood_list, axis=1)  # (n_frames, n_bodyparts)

    return {
        "coords": coords,
        "likelihood": likelihood,
        "bodyparts": bodyparts,
        "scorer": scorer,
        "animal": animal,
    }


def read_dlc(filepath: Union[str, Path], **kwargs) -> BehapyData:
    """
    Read DeepLabCut output and return a BehapyData object.

    Parameters
    ----------
    filepath
        Path to the DLC file (.h5 or .csv).
    **kwargs
        Passed to read_dlc_h5 or read_dlc_csv.

    Returns
    -------
    BehapyData object.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".h5":
        data = read_dlc_h5(filepath, **kwargs)
    elif filepath.suffix == ".csv":
        data = read_dlc_csv(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    n_frames = data["coords"].shape[0]
    n_bodyparts = len(data["bodyparts"])

    # Reshape coords for X: (n_frames, n_bodyparts * 2)
    X = data["coords"].reshape(n_frames, -1)

    # Create obs
    obs = pd.DataFrame(index=[f"frame_{i}" for i in range(n_frames)])
    obs["frame_number"] = np.arange(n_frames)
    obs["animal"] = data["animal"]

    # Create var
    var_list = []
    for bp in data["bodyparts"]:
        var_list.append({"bodypart": bp, "coord_axis": "x"})
        var_list.append({"bodypart": bp, "coord_axis": "y"})
    var = pd.DataFrame(var_list)
    var.index = [f"{d['bodypart']}_{d['coord_axis']}" for d in var_list]

    # Create layers
    # Repeat likelihood for each coordinate axis (x, y) to match var dimension
    likelihood_expanded = np.repeat(data["likelihood"], 2, axis=1)

    layers = {"raw": X.copy(), "likelihood": likelihood_expanded}

    # Create uns
    uns = {
        "software": "deeplabcut",
        "scorer": data["scorer"],
        "filepath": str(filepath),
        "bodyparts": data["bodyparts"],
    }

    return BehapyData(X=X, obs=obs, var=var, layers=layers, uns=uns)
