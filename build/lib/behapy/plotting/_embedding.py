import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional, Union

from .._core._behapydata import BehapyData


def embedding(
    bdata: BehapyData,
    basis: str = "umap",
    color: Optional[Union[str, List[str]]] = None,
    size: float = 5,
    alpha: float = 0.8,
    max_points: Optional[int] = 10000,
    legend_loc: str = "right",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot embedding coordinates.

    Parameters
    ----------
    bdata
        BehapyData object.
    basis
        Embedding basis ('umap', 'tsne', 'pca').
    color
        Key in bdata.obs or bdata.var to color the points.
    size
        Point size.
    alpha
        Transparency.
    max_points
        Maximum number of points to plot. If n_obs > max_points, 
        data will be downsampled. Set to None to disable.
    legend_loc
        Location of the legend.
    ax
        Matplotlib Axes object.
    show
        Whether to show the plot.
    **kwargs
        Additional arguments passed to sns.scatterplot.

    Returns
    -------
    Matplotlib Axes object.
    """
    basis_key = f"X_{basis}"
    if basis_key not in bdata.obsm:
        raise ValueError(f"Basis {basis_key} not found in bdata.obsm.")

    indices = np.arange(bdata.n_obs)
    if max_points is not None and len(indices) > max_points:
        step = len(indices) // max_points
        indices = indices[::step][:max_points]
    
    bdata_subset = bdata[indices]
    coords = bdata_subset.obsm[basis_key]
    df = pd.DataFrame(coords[:, :2], columns=[f"{basis}1", f"{basis}2"], index=bdata_subset.obs_names)

    if color is not None:
        if isinstance(color, str):
            if color in bdata_subset.obs.columns:
                df[color] = bdata_subset.obs[color]
            elif color in bdata_subset.obsm:
                # Support multi-dimensional obsm entries (e.g., speed)
                val = bdata_subset.obsm[color]
                if isinstance(val, pd.DataFrame):
                    df[color] = val.iloc[:, 0].values
                else:
                    df[color] = val[:, 0]
            elif color in bdata_subset.var_names:
                df[color] = bdata_subset[:, color].X.flatten()
            else:
                # Assume it's a constant color
                pass
        else:
            # Assume it's an array-like
            if len(color) == bdata.n_obs:
                df["color"] = np.array(color)[indices]
            else:
                df["color"] = color
            color = "color"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x=f"{basis}1",
        y=f"{basis}2",
        hue=color if isinstance(color, str) and color in df.columns else None,
        palette="viridis" if color and pd.api.types.is_numeric_dtype(df[color]) else None,
        s=size,
        alpha=alpha,
        ax=ax,
        **kwargs,
    )

    if color and color in df.columns:
        if pd.api.types.is_categorical_dtype(df[color]) or not pd.api.types.is_numeric_dtype(df[color]):
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=color)
        else:
            # Colorbar handled by seaborn or manually if needed
            pass

    ax.set_title(f"{basis.upper()} embedding")

    if show:
        plt.show()

    return ax


def umap(bdata: BehapyData, color: Optional[str] = None, **kwargs) -> plt.Axes:
    """Wrapper for embedding with basis='umap'."""
    return embedding(bdata, basis="umap", color=color, **kwargs)


def tsne(bdata: BehapyData, color: Optional[str] = None, **kwargs) -> plt.Axes:
    """Wrapper for embedding with basis='tsne'."""
    return embedding(bdata, basis="tsne", color=color, **kwargs)


def pca(
    bdata: BehapyData, 
    color: Optional[str] = None, 
    components: List[int] = [0, 1], 
    max_points: Optional[int] = 10000,
    **kwargs
) -> plt.Axes:
    """
    Plot PCA coordinates.
    """
    if "X_pca" not in bdata.obsm:
        from ..tools import pca as compute_pca
        compute_pca(bdata)

    indices = np.arange(bdata.n_obs)
    if max_points is not None and len(indices) > max_points:
        step = len(indices) // max_points
        indices = indices[::step][:max_points]
    
    bdata_subset = bdata[indices]
    coords = bdata_subset.obsm["X_pca"][:, components]
    df = pd.DataFrame(coords, columns=[f"PC{components[0]+1}", f"PC{components[1]+1}"], index=bdata_subset.obs_names)

    if color is not None:
        if color in bdata_subset.obs.columns:
            df[color] = bdata_subset.obs[color]
        elif color in bdata_subset.obsm:
            val = bdata_subset.obsm[color]
            if isinstance(val, pd.DataFrame):
                df[color] = val.iloc[:, 0].values
            else:
                df[color] = val[:, 0]

    if "ax" not in kwargs:
        fig, ax = plt.subplots(figsize=(8, 6))
        kwargs["ax"] = ax

    sns.scatterplot(
        data=df,
        x=f"PC{components[0]+1}",
        y=f"PC{components[1]+1}",
        hue=color,
        s=kwargs.get("size", 5),
        alpha=kwargs.get("alpha", 0.8),
        ax=kwargs["ax"],
    )

    kwargs["ax"].set_title("PCA embedding")

    if kwargs.get("show", True):
        plt.show()

    return kwargs["ax"]


def pca_variance_ratio(
    bdata: BehapyData, n_pcs: int = 50, log: bool = False, ax: Optional[plt.Axes] = None, show: bool = True
) -> plt.Axes:
    """
    Plot PCA variance ratio.
    """
    if "pca" not in bdata.uns:
        raise ValueError("PCA has not been computed. Run tl.pca() first.")

    variance_ratio = bdata.uns["pca"]["variance_ratio"][:n_pcs]
    cum_variance_ratio = np.cumsum(variance_ratio)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(1, len(variance_ratio) + 1)
    ax.bar(x, variance_ratio, alpha=0.5, align="center", label="Individual explained variance")
    ax.step(x, cum_variance_ratio, where="mid", label="Cumulative explained variance")

    if log:
        ax.set_yscale("log")

    ax.axhline(y=0.8, color="r", linestyle="--", label="80% threshold")

    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    ax.legend(loc="best")

    if show:
        plt.show()

    return ax
