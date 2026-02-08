import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict

from ..external.pyrat_core import processing as pyrat_proc
from ..external.vame_core.analysis import umap_visualization as vame_vis

def plot_behaviorflow_zones(
    adata,
    x_col: str = 'x',
    y_col: str = 'y',
    zone_key: str = 'zone_stats_zones',
    ax: Optional[plt.Axes] = None,
    show_trajectory: bool = True,
    **kwargs
):
    """
    Plot zones and trajectory (BehaviorFlow style).
    
    Args:
        adata: BehapyData object.
        x_col: Column name for x coordinates.
        y_col: Column name for y coordinates.
        zone_key: Key in adata.uns where zones are stored.
        ax: Matplotlib axes.
        show_trajectory: Whether to plot trajectory.
        **kwargs: Arguments passed to patches.Polygon.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    if show_trajectory:
        if x_col in adata.obs and y_col in adata.obs:
            ax.plot(adata.obs[x_col], adata.obs[y_col], color='gray', alpha=0.5, label='Trajectory')
            
    if zone_key in adata.uns:
        zones = adata.uns[zone_key]
        for name, coords in zones.items():
            if isinstance(coords, pd.DataFrame):
                poly_coords = list(zip(coords['x'], coords['y']))
            elif isinstance(coords, dict) and 'x' in coords:
                poly_coords = list(zip(coords['x'], coords['y']))
            else:
                continue
            
            # Default style if not provided
            if 'facecolor' not in kwargs:
                kwargs['facecolor'] = 'none'
            if 'edgecolor' not in kwargs:
                kwargs['edgecolor'] = 'red'
                
            poly = patches.Polygon(poly_coords, closed=True, alpha=0.5, label=name, **kwargs)
            ax.add_patch(poly)
            
            cx = np.mean([p[0] for p in poly_coords])
            cy = np.mean([p[1] for p in poly_coords])
            ax.text(cx, cy, name, ha='center', va='center', fontsize=10, color='black')
            
    ax.set_aspect('equal')
    return ax

def plot_neg_grids(
    adata,
    x_col: str = 'x_m',
    y_col: str = 'y_m',
    grid_key: str = 'exploration_grids',
    ax: Optional[plt.Axes] = None,
    show_trajectory: bool = True,
    **kwargs
):
    """
    Plot NEG grids.
    
    Args:
        adata: BehapyData object.
        x_col: Column name for x coordinates.
        y_col: Column name for y coordinates.
        grid_key: Key in adata.uns where grids are stored.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    if show_trajectory and x_col in adata.obs and y_col in adata.obs:
        ax.plot(adata.obs[x_col], adata.obs[y_col], color='gray', alpha=0.5, label='Trajectory')
        
    if grid_key in adata.uns:
        grids = adata.uns[grid_key]
        # grids columns: grid_type, band_id, x_min, x_max, y_min, y_max
        
        for _, row in grids.iterrows():
            rect = patches.Rectangle(
                (row['x_min'], row['y_min']),
                row['x_max'] - row['x_min'],
                row['y_max'] - row['y_min'],
                linewidth=1,
                edgecolor='r' if row['grid_type'] == 'closed' else 'b',
                facecolor='none',
                alpha=0.3
            )
            ax.add_patch(rect)
            
    ax.set_aspect('equal')
    return ax

def plot_pyrat_trajectory(
    adata,
    bodypart: str = 'body',
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    **kwargs
):
    """
    Plot trajectory using PyRAT's engine.
    Adapts AnnData to PyRAT's expected input format.
    """
    # Determine columns
    if x_col is None: x_col = f"{bodypart}_x" # Guess
    if y_col is None: y_col = f"{bodypart}_y"
    
    # Try to find them
    if x_col not in adata.obs or y_col not in adata.obs:
        # Fallback to simple x/y if available
        if 'x' in adata.obs and 'y' in adata.obs:
            x_col = 'x'
            y_col = 'y'
            
    if x_col not in adata.obs:
        raise ValueError(f"Could not find coordinates for {bodypart}. Please specify x_col and y_col.")
        
    # Construct fake DLC dataframe
    n_rows = adata.n_obs
    
    # Header rows
    # PyRAT expects: 
    # Row 0: Bodyparts (val, val, val...)
    # Row 1: Coords (x, y, x, y...)
    
    # We create a dataframe with 3 columns: Index, X, Y
    # But PyRAT parses iloc[0][1:] -> bodyparts
    
    header_bp = ['bodyparts', bodypart, bodypart]
    header_co = ['coords', 'x', 'y']
    
    data_x = adata.obs[x_col].values
    data_y = adata.obs[y_col].values
    
    # Create object array to hold strings and floats
    full_arr = np.empty((n_rows + 2, 3), dtype=object)
    full_arr[0] = header_bp
    full_arr[1] = header_co
    full_arr[2:, 0] = np.arange(n_rows)
    full_arr[2:, 1] = data_x
    full_arr[2:, 2] = data_y
    
    df = pd.DataFrame(full_arr)
    
    # Call PyRAT
    pyrat_proc.Trajectory(df, bodypart, **kwargs)

def plot_vame_umap(
    adata,
    basis: str = 'vame',
    label_col: Optional[str] = None,
    **kwargs
):
    """
    Plot VAME UMAP.
    
    Args:
        adata: BehapyData object.
        basis: Basis name (e.g. 'vame' -> X_vame).
        label_col: Column in obs for coloring (clusters).
    """
    if f'X_{basis}' not in adata.obsm:
        raise ValueError(f"Basis X_{basis} not found")
        
    embed = adata.obsm[f'X_{basis}']
    
    if label_col and label_col in adata.obs:
        labels = adata.obs[label_col].values
        # Convert to numeric if possible for coloring, or map unique to ints
        uniques, codes = np.unique(labels, return_inverse=True)
        n_cluster = len(uniques)
        
        # vame_vis.umap_label_vis(file, embed, label, n_cluster, num_points)
        vame_vis.umap_label_vis(None, embed, codes, n_cluster, len(embed))
    else:
        vame_vis.umap_vis(None, embed, len(embed))
