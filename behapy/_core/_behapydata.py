from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData


class BehapyData(AnnData):
    """
    A Scanpy-like framework for analyzing high-dimensional behavioral time series data.
    Wraps AnnData to provide a structured format for pose estimation and behavior data.
    """

    def __init__(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Dict[str, Any]] = None,
        obsm: Optional[Dict[str, Any]] = None,
        varm: Optional[Dict[str, Any]] = None,
        layers: Optional[Dict[str, Any]] = None,
        raw: Optional[Any] = None,
        dtype: str = "float32",
        shape: Optional[tuple] = None,
        filename: Optional[str] = None,
        filemode: Optional[str] = None,
        asview: bool = False,
        **kwargs,
    ):
        super().__init__(
            X=X,
            obs=obs,
            var=var,
            uns=uns,
            obsm=obsm,
            varm=varm,
            layers=layers,
            raw=raw,
            dtype=dtype,
            shape=shape,
            filename=filename,
            filemode=filemode,
            asview=asview,
            **kwargs,
        )
        self._validate()

    def _validate(self):
        """Perform basic validation of the data structure."""
        if self.X is not None:
            if not isinstance(self.X, (np.ndarray, pd.DataFrame)):
                try:
                    import scipy.sparse as sp

                    if not sp.issparse(self.X):
                        raise TypeError(
                            "X must be a numpy array, pandas DataFrame or sparse matrix"
                        )
                except ImportError:
                    raise TypeError("X must be a numpy array or pandas DataFrame")

    @property
    def n_frames(self) -> int:
        """Number of frames (observations)."""
        return self.n_obs

    @property
    def n_features(self) -> int:
        """Number of features (variables)."""
        return self.n_vars

    def __repr__(self) -> str:
        descr = f"BehapyData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}"
        for attr in ["obs", "var", "uns", "obsm", "varm", "layers"]:
            keys = list(getattr(self, attr).keys())
            if len(keys) > 0:
                descr += f"\n    {attr}: {', '.join(map(str, keys))}"
        return descr

    def copy(self) -> "BehapyData":
        """Return a copy of the BehapyData object."""
        new = super().copy()
        new.__class__ = BehapyData
        return new

    def __getitem__(self, index) -> "BehapyData":
        """Return a sliced view of the BehapyData object."""
        new = super().__getitem__(index)
        new.__class__ = BehapyData
        return new
