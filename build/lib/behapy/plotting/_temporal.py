from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .._core._behapydata import BehapyData

def transition_matrix(
    bdata: BehapyData, 
    key: str = "leiden", 
    figsize: Tuple[int, int] = (10, 8), 
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    comparison_bdata: Optional[BehapyData] = None,
    show_significance: bool = True,
    sig_threshold: float = 0.05
) -> plt.Axes:
    """
    Plot cluster transition matrix as a heatmap.
    
    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing cluster labels.
    figsize
        Figure size.
    save
        Path to save the plot.
    ax
        Pre-existing axes for plotting.
    comparison_bdata
        Optional second BehapyData object for comparison.
    show_significance
        If True, show significance markers (*, **, ***) when comparison_bdata is provided.
    sig_threshold
        Significance threshold (default 0.05).
        
    Returns
    -------
    matplotlib.axes.Axes object.
    """
    trans_key = f"{key}_transitions"
    if trans_key not in bdata.uns:
        from ..tools._temporal import compute_transitions
        compute_transitions(bdata, key=key)
        
    matrix = bdata.uns[trans_key]
    labels = bdata.uns[f"{key}_transition_labels"]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    annot = True
    if comparison_bdata is not None:
        from ..tools._statistics import test_transition_matrix
        results = test_transition_matrix(bdata, comparison_bdata, key=key)
        
        # Create custom annotation matrix with significance markers
        annot_matrix = []
        for i in range(matrix.shape[0]):
            row = []
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                # Get p-value from results
                p_val = results[(results['from_behavior'] == labels[i]) & 
                                (results['to_behavior'] == labels[j])]['pvalue'].values[0]
                
                sig_marker = ""
                if show_significance:
                    if p_val < 0.001: sig_marker = "***"
                    elif p_val < 0.01: sig_marker = "**"
                    elif p_val < 0.05: sig_marker = "*"
                
                row.append(f"{val:.2f}{sig_marker}")
            annot_matrix.append(row)
        annot = np.array(annot_matrix)

    sns.heatmap(
        matrix, 
        annot=annot, 
        fmt="" if comparison_bdata is not None else ".2f", 
        cmap="viridis", 
        xticklabels=labels, 
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(f"Transition Matrix: {key}")
    ax.set_xlabel("To Cluster")
    ax.set_ylabel("From Cluster")
    
    if save:
        plt.savefig(save, bbox_inches='tight')
        
    return ax

def ethogram(
    bdata: BehapyData, 
    key: str = "leiden", 
    start: int = 0, 
    end: Optional[int] = None, 
    figsize: Tuple[int, int] = (15, 3), 
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Horizontal timeline colored by cluster.
    
    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing cluster labels.
    start
        Start frame index.
    end
        End frame index.
    figsize
        Figure size.
    save
        Path to save the plot.
    ax
        Pre-existing axes for plotting.
        
    Returns
    -------
    matplotlib.axes.Axes object.
    """
    if key not in bdata.obs:
        raise ValueError(f"Key '{key}' not found in bdata.obs.")
        
    if end is None:
        end = bdata.n_obs
        
    subset = bdata.obs[key].iloc[start:end].values
    unique_labels = sorted(bdata.obs[key].unique())
    n_clusters = len(unique_labels)
    
    # Map labels to integers for plotting
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    numeric_subset = np.array([label_to_idx[l] for l in subset])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    # Create ethogram using imshow
    ax.imshow(
        numeric_subset.reshape(1, -1), 
        aspect="auto", 
        cmap="tab20", 
        interpolation="nearest"
    )
    
    ax.set_yticks([])
    ax.set_xlabel("Frame Index")
    ax.set_title(f"Ethogram: {key} (Frames {start}-{end})")
    
    if save:
        plt.savefig(save, bbox_inches='tight')
        
    return ax

def bout_duration_distribution(
    bdata: BehapyData, 
    key: str = "leiden", 
    clusters: Optional[List[Union[str, int]]] = None, 
    figsize: Tuple[int, int] = (12, 6), 
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot bout duration distribution per cluster.
    
    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing cluster labels.
    clusters
        List of clusters to include. If None, all are included.
    figsize
        Figure size.
    save
        Path to save the plot.
    ax
        Pre-existing axes for plotting.
        
    Returns
    -------
    matplotlib.axes.Axes object.
    """
    bout_key = f"{key}_bouts"
    if bout_key not in bdata.uns:
        from ..tools._temporal import detect_bouts
        detect_bouts(bdata, key=key)
        
    bouts = bdata.uns[bout_key]
    if not bouts:
        print(f"No bouts found for key '{key}'.")
        return ax
        
    df_bouts = pd.DataFrame(bouts)
    
    if clusters is not None:
        df_bouts = df_bouts[df_bouts["cluster"].isin(clusters)]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    sns.boxplot(data=df_bouts, x="cluster", y="duration", ax=ax, palette="tab20")
    sns.stripplot(data=df_bouts, x="cluster", y="duration", color=".3", size=3, alpha=0.5, ax=ax)
    
    ax.set_title(f"Bout Duration Distribution: {key}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Duration (frames)")
    
    if save:
        plt.savefig(save, bbox_inches='tight')
        
    return ax
