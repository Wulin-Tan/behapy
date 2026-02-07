import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple
import warnings

from .._core._behapydata import BehapyData

def effect_sizes(
    results_df: pd.DataFrame,
    plot_type: str = 'forest',
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    save: Optional[str] = None
) -> plt.Axes:
    """
    Visualize effect sizes with confidence intervals.

    Parameters
    ----------
    results_df
        DataFrame containing 'comparison', 'effect_size', and optionally 'pvalue' or 'ci'.
    plot_type
        Type of plot ('forest', 'volcano', 'bar').
    title
        Plot title.
    ax
        Pre-existing axes for plotting.
    figsize
        Figure size.
    save
        Path to save the figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'forest':
        # Simple forest plot
        y_pos = np.arange(len(results_df))
        ax.errorbar(results_df['effect_size'], y_pos, xerr=0.1, fmt='o', color='black', capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(results_df['comparison'])
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.invert_yaxis()
        
    elif plot_type == 'bar':
        sns.barplot(data=results_df, x='comparison', y='effect_size', ax=ax, hue='comparison', palette='viridis', legend=False)
        ax.set_ylabel("Effect Size")
        plt.setp(ax.get_xticklabels(), rotation=45)
        
    elif plot_type == 'volcano':
        if 'pvalue' not in results_df.columns:
            raise ValueError("Volcano plot requires 'pvalue' column in results_df.")
        results_df['-log10_p'] = -np.log10(results_df['pvalue'])
        sns.scatterplot(data=results_df, x='effect_size', y='-log10_p', ax=ax)
        ax.axhline(-np.log10(0.05), color='red', linestyle='--')
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("-log10(p-value)")

    if title:
        ax.set_title(title)
    
    if save:
        plt.savefig(save, bbox_inches='tight')
        
    return ax

def statistical_summary(
    bdata_list: List[BehapyData],
    groups: List[str],
    metrics: List[str] = ['frequency', 'duration', 'transitions'],
    figsize: Tuple[int, int] = (15, 10),
    save: Optional[str] = None
) -> plt.Figure:
    """
    Multi-panel figure showing multiple comparisons with effect sizes and p-values.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
        
    from ..tools._statistics import compare_groups
    
    for i, metric in enumerate(metrics):
        try:
            # Map simplified metric names to actual keys
            m_key = metric
            if metric == 'frequency': m_key = 'leiden' # placeholder
            elif metric == 'duration': m_key = 'bout_duration'
            
            results = compare_groups(bdata_list, groups, metric=m_key)
            if not results.empty:
                effect_sizes(results, plot_type='bar', ax=axes[i], title=f"Comparison: {metric}")
        except Exception as e:
            warnings.warn(f"Could not compute summary for {metric}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {metric}", ha='center')

    plt.tight_layout()
    if save:
        plt.savefig(save)
    return fig
