from typing import Optional
import pandas as pd
from .neg_core import analysis as neg_analysis

def analyze_exploration(
    adata,
    grid_size: float = 2.0,
    x_col: str = 'x_m',
    y_col: str = 'y_m',
    in_open_col: str = 'InOpen',
    in_closed_col: str = 'InClosed',
    in_centre_col: str = 'InCentre',
    key_added: str = 'exploration'
):
    """
    Analyze grid exploration using NEG algorithm.
    
    Args:
        adata: BehapyData object.
        x_col, y_col: Corrected coordinate columns.
        in_open_col, in_closed_col, in_centre_col: Zone boolean columns.
    """
    # Extract DataFrame
    required = [x_col, y_col, in_open_col, in_closed_col, in_centre_col]
    for col in required:
        if col not in adata.obs:
            raise ValueError(f"Column {col} not found in adata.obs")
            
    df = adata.obs[required].copy()
    # Rename to what NEG expects
    df = df.rename(columns={
        x_col: 'x_m',
        y_col: 'y_m',
        in_open_col: 'InOpen',
        in_closed_col: 'InClosed',
        in_centre_col: 'InCentre'
    })
    
    # Create grids
    grids = neg_analysis.create_complete_arm_grids(df, grid_size)
    
    # Calculate exploration
    res = neg_analysis.calculate_grid_exploration(df, grids)
    
    # Store results
    # Res is long format (multiple rows per timepoint due to grid_types).
    
    # If successful, we might have grid_type column
    if 'grid_type' in res.columns:
        total_res = res[res['grid_type'] == 'total']
        # We could add summary stats to obs here if we align index
    
    adata.uns[f'{key_added}_grids'] = grids
    adata.uns[f'{key_added}_results'] = res
    
    return adata
