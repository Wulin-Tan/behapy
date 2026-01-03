from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._core._behapydata import BehapyData


def trajectory(
    bdata: BehapyData,
    bodypart: str,
    color_by: str = "time",
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_points: Optional[int] = 5000,
    alpha: float = 0.8,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot the trajectory of a specified bodypart.

    Parameters
    ----------
    bdata
        BehapyData object.
    bodypart
        Name of the bodypart to plot.
    color_by
        Key in bdata.obs or 'time' to color the trajectory.
    start
        Starting frame index.
    end
        Ending frame index.
    max_points
        Maximum number of points to plot. If n_obs > max_points, 
        data will be downsampled. Set to None to disable.
    alpha
        Transparency of the points.
    ax
        Matplotlib Axes object. If None, a new figure is created.
    show
        Whether to show the plot.
    **kwargs
        Additional arguments passed to plt.scatter.

    Returns
    -------
    Matplotlib Axes object.
    """
    # Subset frames if requested
    start = start or 0
    end = end or bdata.n_obs
    
    indices = np.arange(start, end)
    
    # Downsample if necessary
    if max_points is not None and len(indices) > max_points:
        step = len(indices) // max_points
        indices = indices[::step][:max_points]
        
    bdata_subset = bdata[indices]

    # Get coordinates for the bodypart
    idx = np.where(bdata.var["bodypart"] == bodypart)[0]
    if len(idx) == 0:
        raise ValueError(f"Bodypart {bodypart} not found.")

    # Assuming 2D for now. idx[0] is x, idx[1] is y
    x = bdata_subset.X[:, idx[0]]
    y = bdata_subset.X[:, idx[1]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if color_by == "time":
        c = indices
        label = "Frame Number"
    elif color_by in bdata_subset.obs.columns:
        c = bdata_subset.obs[color_by]
        label = color_by
    elif color_by in bdata_subset.obsm:
        # Handle speed/acceleration if they exist
        if f"speed_{bodypart}" in bdata_subset.obsm[color_by].columns:
            c = bdata_subset.obsm[color_by][f"speed_{bodypart}"]
        else:
            c = bdata_subset.obsm[color_by].iloc[:, 0]
        label = color_by
    else:
        c = None
        label = None

    # Use a more efficient way to plot if color is constant
    if c is None:
        ax.plot(x, y, alpha=alpha, **kwargs)
    else:
        scatter = ax.scatter(x, y, c=c, alpha=alpha, **kwargs)
        if not isinstance(c.iloc[0] if isinstance(c, pd.Series) else c[0], str):
            plt.colorbar(scatter, ax=ax, label=label)

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Trajectory of {bodypart}")

    if show:
        plt.show()

    return ax


def heatmap(
    bdata: BehapyData,
    bodypart: str,
    bins: int = 50,
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Create a 2D heatmap of bodypart positions.

    Parameters
    ----------
    bdata
        BehapyData object.
    bodypart
        Name of the bodypart to plot.
    bins
        Number of bins for the histogram.
    cmap
        Colormap.
    ax
        Matplotlib Axes object. If None, a new figure is created.
    show
        Whether to show the plot.

    Returns
    -------
    Matplotlib Axes object.
    """
    idx = np.where(bdata.var["bodypart"] == bodypart)[0]
    if len(idx) == 0:
        raise ValueError(f"Bodypart {bodypart} not found.")

    x = bdata.X[:, idx[0]]
    y = bdata.X[:, idx[1]]

    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    h = ax.hist2d(x, y, bins=bins, cmap=cmap)
    plt.colorbar(h[3], ax=ax, label="Density")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Position Heatmap: {bodypart}")

    if show:
        plt.show()

    return ax
