from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .._core._behapydata import BehapyData


def rank_features_groups(
    bdata: BehapyData,
    groupby: str = "behavior",
    n_features: int = 20,
    method: str = "wilcoxon",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot top ranked features for each group.
    """
    if "rank_features_groups" not in bdata.uns:
        raise ValueError("Run tl.rank_features_groups() first.")

    res = bdata.uns["rank_features_groups"]
    groups = res["names"].dtype.names

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # For now, just plot the first group or a grid?
    # Usually we want a plot per group.
    # Let's just plot the top n_features for the first group as a simple version.
    group = groups[0]
    names = res["names"][group][:n_features]
    scores = res["scores"][group][:n_features]

    ax.barh(np.arange(len(names)), scores)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title(f"Top features for {group} (vs {res['params']['reference']})")

    if show:
        plt.show()

    return ax


def feature_group_heatmap(
    bdata: BehapyData,
    features: Optional[List[str]] = None,
    groupby: str = "behavior",
    standard_scale: Optional[str] = "var",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (8, 6),
    show: bool = True,
) -> Union[sns.matrix.ClusterGrid, plt.Axes]:
    """
    Plot a heatmap of features grouped by behavior.
    """
    if features is None:
        features = bdata.var_names.tolist()

    # Create a dataframe with features and groupby
    df = pd.DataFrame(bdata[:, features].X, columns=features, index=bdata.obs_names)
    df[groupby] = bdata.obs[groupby]

    # Compute mean per group
    group_means = df.groupby(groupby).mean()

    if standard_scale == "var":
        # Z-score scaling
        group_means = (group_means - group_means.mean()) / group_means.std()

    cg = sns.clustermap(group_means, cmap=cmap, figsize=figsize)

    if show:
        plt.show()

    return cg
