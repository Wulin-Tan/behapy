from pathlib import Path
from typing import Union

from .._core._behapydata import BehapyData
from ._dlc import read_dlc
from ._sleap import read_sleap


def detect_software(filepath: Union[str, Path]) -> str:
    """
    Examine file structure/headers to auto-detect software.

    Returns
    -------
    'deeplabcut', 'sleap', 'anipose', 'c3d', or 'unknown'
    """
    filepath = Path(filepath)
    if filepath.suffix == ".h5":
        # Could be DLC or SLEAP
        import h5py

        try:
            with h5py.File(filepath, "r") as f:
                if "tracks" in f.keys() and "node_names" in f.keys():
                    return "sleap"
                # DLC H5 files are usually pandas HDF5
                if "df_with_missing" in f.keys():
                    return "deeplabcut"
        except Exception:
            pass

        # Try reading as pandas HDF5 (DLC)
        try:
            import pandas as pd

            pd.read_hdf(filepath)
            return "deeplabcut"
        except Exception:
            pass

    elif filepath.suffix == ".csv":
        # Likely DLC
        import pandas as pd

        try:
            df = pd.read_csv(filepath, nrows=3, header=None)
            if "scorer" in df.values:
                return "deeplabcut"
        except Exception:
            pass

    return "unknown"


def read(filepath: Union[str, Path], software: str = "auto", **kwargs) -> BehapyData:
    """
    Read behavioral data from various software formats.

    Parameters
    ----------
    filepath
        Path to the data file.
    software
        Software name ('deeplabcut', 'sleap', 'auto').
    **kwargs
        Additional arguments passed to specific readers.

    Returns
    -------
    BehapyData object.
    """
    if software == "auto":
        software = detect_software(filepath)

    if software == "deeplabcut":
        return read_dlc(filepath, **kwargs)
    elif software == "sleap":
        return read_sleap(filepath, **kwargs)
    else:
        raise ValueError(f"Unknown or unsupported software: {software}")
