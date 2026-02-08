from typing import List, Optional, Union
import numpy as np
import pandas as pd
from .behaviorflow_core import analysis as bf_analysis
from .behaviorflow_core.io import TrackingData

def calculate_movement(
    adata,
    x_col: str = 'x',
    y_col: str = 'y',
    movement_cutoff: float = 5.0,
    integration_period: int = 5,
    key_added: str = 'movement'
):
    """
    Calculate movement statistics using BehaviorFlow algorithm.
    
    Args:
        adata: BehapyData or AnnData object.
        x_col: Column name in adata.obs or adata.to_df() for x coordinates.
               If not found, checks adata.X if it has 2 columns.
        y_col: Column name for y coordinates.
        movement_cutoff: Speed threshold.
        integration_period: Smoothing window.
        key_added: Prefix for added columns in adata.obs.
        
    Returns:
        Updates adata.obs with '{key_added}_speed', '{key_added}_acceleration', '{key_added}_is_moving'.
    """
    # Extract data
    if x_col in adata.obs and y_col in adata.obs:
        x = adata.obs[x_col]
        y = adata.obs[y_col]
    elif adata.n_vars >= 2:
        # Assuming X contains coordinates
        if isinstance(adata.X, pd.DataFrame):
            x = adata.X.iloc[:, 0]
            y = adata.X.iloc[:, 1]
        else:
            x = adata.X[:, 0]
            y = adata.X[:, 1]
    else:
        raise ValueError(f"Could not find coordinates. Provide valid x_col/y_col or ensure X has coordinates.")
        
    # Create temp dataframe for TrackingData
    # BehaviorFlow expects a dict of DataFrames per point
    # We'll treat the whole object as one "point" (e.g. "body")
    
    df = pd.DataFrame({'x': x, 'y': y})
    frames = np.arange(len(df))
    fps = adata.uns.get('fps', 30) # Default to 30 if not set
    
    data_dict = {'body': df}
    
    td = TrackingData(data_dict, frames, fps, "behapy_data")
    
    # Run analysis
    td = bf_analysis.calculate_movement(td, movement_cutoff, integration_period)
    
    # Extract results
    res_df = td.data['body']
    
    adata.obs[f'{key_added}_speed'] = res_df['speed'].values
    adata.obs[f'{key_added}_acceleration'] = res_df['acceleration'].values
    adata.obs[f'{key_added}_is_moving'] = res_df['is_moving'].values
    
    return adata

def zone_analysis(
    adata,
    zones: dict,
    x_col: str = 'x',
    y_col: str = 'y',
    movement_key: str = 'movement',
    key_added: str = 'zone_stats'
):
    """
    Run zone analysis.
    
    Args:
        zones: Dictionary of zone definitions. 
               Format: {'zone_name': pd.DataFrame({'x':..., 'y':...}) or dict}
        movement_key: Key prefix used in calculate_movement (default: 'movement').
    """
    # Setup TrackingData
    if x_col in adata.obs and y_col in adata.obs:
        x = adata.obs[x_col]
        y = adata.obs[y_col]
    else:
        # Try to use columns from previous movement analysis or X
        raise ValueError("Coordinates not found.")
        
    df = pd.DataFrame({'x': x, 'y': y})
    
    # If movement calculated, include it
    speed_col = f'{movement_key}_speed'
    acc_col = f'{movement_key}_acceleration'
    moving_col = f'{movement_key}_is_moving'
    
    if speed_col in adata.obs:
        # TODO: Add logic to pass existing movement stats to TrackingData if needed
        # For now, bf_analysis.zone_report calculates stats based on tracking data
        pass
        
    frames = np.arange(len(df))
    fps = adata.uns.get('fps', 30)
    
    td = TrackingData({'body': df}, frames, fps, "behapy_data")
    td.zones = zones
    
    # Store zones in uns for plotting
    adata.uns[f'{key_added}_zones'] = zones

    # Run analysis for each zone
    reports = {}
    for zone_name in zones:
        rep = bf_analysis.zone_report(td, 'body', zone_name)
        reports.update(rep)
        
    if 'zone_reports' not in adata.uns:
        adata.uns['zone_reports'] = {}
    
    adata.uns['zone_reports'].update(reports)
    
    return adata
