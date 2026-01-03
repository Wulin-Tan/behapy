from pathlib import Path
from typing import Union

import pandas as pd

from .._core._behapydata import BehapyData


def write_h5ad(bdata: BehapyData, filepath: Union[str, Path]):
    """
    Save BehapyData object to H5AD file.
    """
    bdata.write_h5ad(filepath)


def write_csv(bdata: BehapyData, filepath: Union[str, Path]):
    """
    Export coordinate matrix to CSV.
    """
    df = bdata.to_df()
    df.to_csv(filepath)
