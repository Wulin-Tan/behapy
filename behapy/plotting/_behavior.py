from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .._core._behapydata import BehapyData


def _extract_bouts(labels: np.ndarray) -> List[Tuple[str, int, int]]:
    """
    Extract continuous bouts from a sequence of labels.

    Returns
    -------
    List of (label, start_frame, duration)
    """
    if len(labels) == 0:
        return []

    bouts = []
    curr_label = labels[0]
    curr_start = 0

    for i in range(1, len(labels)):
        if labels[i] != curr_label:
            bouts.append((curr_label, curr_start, i - curr_start))
            curr_label = labels[i]
            curr_start = i

    # Add last bout
    bouts.append((curr_label, curr_start, len(labels) - curr_start))

    return bouts


def ethogram(
    bdata: BehapyData,
    behavior_key: str = "behavior",
    start: Optional[int] = None,
    end: Optional[int] = None,
    fps: int = 30,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Create an ethogram plot.

    Parameters
    ----------
    bdata
        BehapyData object.
    behavior_key
        Column in bdata.obs containing behavior labels.
    start
        Start frame.
    end
        End frame.
    fps
        Frames per second.
    ax
        Matplotlib Axes.
    show
        Whether to show the plot.
    **kwargs
        Additional arguments passed to plt.barh.
    """
    if behavior_key not in bdata.obs.columns:
        raise ValueError(f"Behavior key {behavior_key} not found in bdata.obs.")

    labels = bdata.obs[behavior_key].values
    if start is not None or end is not None:
        start = start or 0
        end = end or len(labels)
        labels = labels[start:end]
    else:
        start = 0

    bouts = _extract_bouts(labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))

    # Unique behaviors and their colors
    unique_behaviors = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_behaviors))
    color_map = {beh: colors(i) for i, beh in enumerate(unique_behaviors)}

    for beh, bout_start, duration in bouts:
        ax.barh(
            0,
            duration / fps,
            left=(start + bout_start) / fps,
            color=color_map[beh],
            **kwargs,
        )

    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Ethogram: {behavior_key}")

    # Custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=color_map[beh], lw=4, label=beh) for beh in unique_behaviors
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.1, 1.0))

    if show:
        plt.show()

    return ax


def behavior_pie(
    bdata: BehapyData, behavior_key: str = "behavior", ax: Optional[plt.Axes] = None, show: bool = True
) -> plt.Axes:
    """
    Create a pie chart of behavior distribution.
    """
    if behavior_key not in bdata.obs.columns:
        raise ValueError(f"Behavior key {behavior_key} not found in bdata.obs.")

    counts = bdata.obs[behavior_key].value_counts()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title(f"Behavior Distribution: {behavior_key}")

    if show:
        plt.show()

    return ax


def feature_time_heatmap(
    bdata: BehapyData,
    features: Optional[List[str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_points: Optional[int] = 5000,
    use_rep: Optional[str] = None,
    standard_scale: Optional[str] = "var",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a heatmap of features over time.

    Parameters
    ----------
    bdata
        BehapyData object.
    features
        List of features to plot. If None, uses all features.
    start
        Start frame index.
    end
        End frame index.
    max_points
        Maximum number of time points to plot. If n_obs > max_points, 
        data will be downsampled. Set to None to disable.
    use_rep
        Which representation to use. If None, uses bdata.X.
    standard_scale
        Whether to scale variables ('var') or observations ('obs').
    cmap
        Colormap.
    figsize
        Figure size.
    ax
        Matplotlib Axes.
    show
        Whether to show the plot.
    """
    if features is None:
        if use_rep:
            features = [f"feat_{i}" for i in range(bdata.obsm[use_rep].shape[1])]
        else:
            features = bdata.var_names.tolist()

    if use_rep:
        X = bdata.obsm[use_rep]
    else:
        X = bdata.X

    start = start or 0
    end = end or X.shape[0]
    
    indices = np.arange(start, end)
    if max_points is not None and len(indices) > max_points:
        step = len(indices) // max_points
        indices = indices[::step][:max_points]
        
    X_subset = X[indices]

    df = pd.DataFrame(X_subset, columns=features)

    if standard_scale == "var":
        df = (df - df.mean()) / df.std()
    elif standard_scale == "obs":
        df = df.div(df.sum(axis=1), axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(df.T, cmap=cmap, ax=ax, cbar_kws={"label": "Scaled Value"})

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Features")
    ax.set_title("Feature Heatmap Over Time")

    if show:
        plt.show()

    return ax


def time_series(
    bdata: BehapyData,
    key: str,
    use_rep: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_points: Optional[int] = 5000,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot a time series of a feature or observation metadata.

    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs, bdata.obsm, or bdata.var_names.
    use_rep
        If key is in bdata.obsm, which column index or name to use.
    start
        Start frame index.
    end
        End frame index.
    max_points
        Maximum number of points to plot. If n_obs > max_points, 
        data will be downsampled. Set to None to disable.
    ax
        Matplotlib Axes.
    show
        Whether to show the plot.
    **kwargs
        Additional arguments passed to plt.plot.
    """
    if key in bdata.obs.columns:
        y = bdata.obs[key].values
    elif key in bdata.var_names:
        y = bdata[:, key].X.flatten()
    elif key in bdata.obsm:
        if use_rep is not None:
            if isinstance(use_rep, str) and use_rep in bdata.obsm[key].columns:
                y = bdata.obsm[key][use_rep].values
            else:
                y = bdata.obsm[key].iloc[:, use_rep].values
        else:
            y = bdata.obsm[key].iloc[:, 0].values
    else:
        raise ValueError(f"Key {key} not found in bdata.obs, bdata.obsm, or bdata.var_names.")

    start = start or 0
    end = end or len(y)
    
    indices = np.arange(start, end)
    if max_points is not None and len(indices) > max_points:
        step = len(indices) // max_points
        indices = indices[::step][:max_points]
        
    y_subset = y[indices]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(indices, y_subset, **kwargs)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel(key)
    ax.set_title(f"{key} over time")

    if show:
        plt.show()

    return ax


def bout_distribution(
    bdata: BehapyData,
    behavior_key: str = "behavior",
    bins: int = 30,
    log: bool = False,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot distribution of bout durations per behavior.
    """
    if behavior_key not in bdata.obs.columns:
        raise ValueError(f"Behavior key {behavior_key} not found in bdata.obs.")

    labels = bdata.obs[behavior_key].values
    bouts = _extract_bouts(labels)

    df = pd.DataFrame(bouts, columns=["behavior", "start", "duration"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data=df, x="duration", hue="behavior", bins=bins, log_scale=log, ax=ax, multiple="stack")

    ax.set_xlabel("Bout Duration (frames)")
    ax.set_title(f"Bout Duration Distribution: {behavior_key}")

    if show:
        plt.show()

    return ax
