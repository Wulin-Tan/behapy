import pandas as pd
import numpy as np

def safe_seq(start, end, step):
    if pd.isna(start) or pd.isna(end) or pd.isna(step):
        return np.array([])
    if start == end:
        return np.array([start])
    if (end > start and step <= 0) or (end < start and step >= 0):
        return np.array([])
    
    vals = []
    curr = start
    epsilon = 1e-10
    if step > 0:
        while curr <= end + epsilon:
            vals.append(curr)
            curr += step
    else:
        while curr >= end - epsilon:
            vals.append(curr)
            curr += step
    return np.array(vals)

def make_bands(depths, min_val, max_val, axis='y'):
    if len(depths) < 2 or pd.isna(min_val) or pd.isna(max_val):
        return pd.DataFrame()
        
    depth_start = depths[:-1]
    depth_end = depths[1:]
    
    df = pd.DataFrame({
        'depth_start': depth_start,
        'depth_end': depth_end,
    })
    
    df['depth_low'] = df[['depth_start', 'depth_end']].min(axis=1)
    df['depth_high'] = df[['depth_start', 'depth_end']].max(axis=1)
    
    if axis == 'y': 
        df['x_min'] = min_val
        df['x_max'] = max_val
        df['y_min'] = df['depth_low']
        df['y_max'] = df['depth_high']
    else: 
        df['y_min'] = min_val
        df['y_max'] = max_val
        df['x_min'] = df['depth_low']
        df['x_max'] = df['depth_high']
        
    return df[['x_min', 'x_max', 'y_min', 'y_max']]

def create_complete_arm_grids(data, grid_size=2):
    """
    Creates grid bands for open and closed arms based on data distribution.
    """
    open_data = data[(data['InOpen'] == 1) & (data['InCentre'] == 0)].dropna(subset=['x_m', 'y_m'])
    closed_data = data[(data['InClosed'] == 1) & (data['InCentre'] == 0)].dropna(subset=['x_m', 'y_m'])
    center_data = data[data['InCentre'] == 1].dropna(subset=['x_m', 'y_m'])
    
    if len(center_data) == 0:
        return pd.DataFrame()
        
    cx_min, cx_max = center_data['x_m'].min(), center_data['x_m'].max()
    cy_min, cy_max = center_data['y_m'].min(), center_data['y_m'].max()
    
    open_bands_list = []
    if len(open_data) > 0:
        ox_min, ox_max = open_data['x_m'].min(), open_data['x_m'].max()
        oy_max = open_data['y_m'].max()
        oy_min = open_data['y_m'].min()
        
        pos_depths = safe_seq(cy_max, oy_max, grid_size)
        neg_depths = safe_seq(cy_min, oy_min, -grid_size)
        
        b1 = make_bands(pos_depths, ox_min, ox_max, axis='y')
        b2 = make_bands(neg_depths, ox_min, ox_max, axis='y')
        
        open_bands_list.append(b1)
        open_bands_list.append(b2)
        
    open_bands = pd.concat(open_bands_list, ignore_index=True) if open_bands_list else pd.DataFrame()
    if not open_bands.empty:
        open_bands['grid_type'] = 'open'
        
    closed_bands_list = []
    if len(closed_data) > 0:
        cy_min_c, cy_max_c = closed_data['y_m'].min(), closed_data['y_m'].max()
        cx_max_c = closed_data['x_m'].max()
        cx_min_c = closed_data['x_m'].min()
        
        pos_depths = safe_seq(cx_max, cx_max_c, grid_size)
        neg_depths = safe_seq(cx_min, cx_min_c, -grid_size)
        
        b1 = make_bands(pos_depths, cy_min_c, cy_max_c, axis='x')
        b2 = make_bands(neg_depths, cy_min_c, cy_max_c, axis='x')
        
        closed_bands_list.append(b1)
        closed_bands_list.append(b2)
        
    closed_bands = pd.concat(closed_bands_list, ignore_index=True) if closed_bands_list else pd.DataFrame()
    if not closed_bands.empty:
        closed_bands['grid_type'] = 'closed'
        
    all_bands = pd.concat([open_bands, closed_bands], ignore_index=True)
    
    if all_bands.empty:
        return all_bands
        
    all_bands['band_id'] = range(1, len(all_bands) + 1)
    
    return all_bands

def calculate_grid_exploration(trial_data, grid_data):
    """
    Calculates exploration metrics based on grid traversal.
    """
    if grid_data.empty:
        return trial_data
    
    results = []
    
    total_grid = grid_data.copy()
    total_grid['grid_type'] = 'total'
    total_grid['band_id'] = range(1, len(total_grid) + 1)
    
    full_grid_data = pd.concat([grid_data, total_grid], ignore_index=True)
    
    for g_type, g_df in full_grid_data.groupby('grid_type'):
        res_df = trial_data.copy()
        res_df['grid_type'] = g_type
        
        visited = set()
        total_bands = len(g_df)
        
        bands_arr = g_df[['x_min', 'x_max', 'y_min', 'y_max', 'band_id']].values
        
        x = trial_data['x_m'].values
        y = trial_data['y_m'].values
        
        point_band_ids = np.zeros(len(trial_data), dtype=int)
        
        for bx_min, bx_max, by_min, by_max, bid in bands_arr:
            mask = (x >= bx_min) & (x <= bx_max) & (y >= by_min) & (y <= by_max)
            point_band_ids[mask] = int(bid)
            
        visited_mask = np.zeros(len(trial_data), dtype=bool)
        current_visited = set()
        
        for i, bid in enumerate(point_band_ids):
            if bid != 0 and bid not in current_visited:
                current_visited.add(bid)
                visited_mask[i] = True
                
        res_df['is_new_visit'] = visited_mask
        res_df['cumulative_visits'] = res_df['is_new_visit'].cumsum()
        res_df['exploration_percentage'] = (res_df['cumulative_visits'] / total_bands * 100) if total_bands > 0 else 0
        
        results.append(res_df)
        
    return pd.concat(results, ignore_index=True)
