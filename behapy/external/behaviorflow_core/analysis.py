import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
try:
    from matplotlib.path import Path
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def calculate_movement(tracking_data, movement_cutoff, integration_period):
    """
    Calculates speed, acceleration, and movement status for all body parts.
    
    Args:
        tracking_data (TrackingData): The tracking data object.
        movement_cutoff (float): Speed threshold for movement (in distance units per second).
        integration_period (int): Window size for smoothing movement detection (frames).
        
    Returns:
        TrackingData: The updated tracking data object with added columns.
    """
    fps = tracking_data.fps
    
    for point_name, df in tracking_data.data.items():
        df['delta_x'] = df['x'].diff().fillna(0)
        df['delta_y'] = df['y'].diff().fillna(0)
        df['speed'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
        df['acceleration'] = df['speed'].diff().fillna(0)
        
        threshold_per_frame = movement_cutoff / fps
        is_moving_raw = df['speed'] > threshold_per_frame
        
        if integration_period > 1:
            df['is_moving'] = is_moving_raw.rolling(window=integration_period, center=True, min_periods=1).mean() > 0.5
        else:
            df['is_moving'] = is_moving_raw
            
        tracking_data.data[point_name] = df
        
    tracking_data.integration_period = integration_period
    return tracking_data

def is_in_zone(tracking_data, point_name, zone_names, invert=False):
    """
    Checks if a point is in the specified zone(s).
    """
    if not hasattr(tracking_data, 'zones') or not tracking_data.zones:
        return None
        
    if point_name not in tracking_data.data:
        return None
        
    df = tracking_data.data[point_name]
    
    if isinstance(zone_names, str):
        zone_names = [zone_names]
        
    in_zone = np.zeros(len(df), dtype=bool)
    
    valid_mask = ~df['x'].isna() & ~df['y'].isna()
    valid_points = df.loc[valid_mask, ['x', 'y']].values
    
    for z_name in zone_names:
        if z_name not in tracking_data.zones:
            continue
            
        poly_coords = tracking_data.zones[z_name]
        
        if isinstance(poly_coords, pd.DataFrame):
            coords = list(zip(poly_coords['x'], poly_coords['y']))
        elif isinstance(poly_coords, dict) and 'x' in poly_coords:
             coords = list(zip(poly_coords['x'], poly_coords['y']))
        else:
            continue
            
        if HAS_MATPLOTLIB:
            path = Path(coords)
            current_in_zone = np.zeros(len(df), dtype=bool)
            if len(valid_points) > 0:
                current_in_zone[valid_mask] = path.contains_points(valid_points)
            in_zone = in_zone | current_in_zone
        else:
            poly = Polygon(coords)
            current_in_zone = []
            for _, row in df.iterrows():
                if pd.isna(row['x']) or pd.isna(row['y']):
                    current_in_zone.append(False)
                else:
                    current_in_zone.append(poly.contains(Point(row['x'], row['y'])))
            in_zone = in_zone | np.array(current_in_zone)

    if invert:
        in_zone = ~in_zone
        
    return in_zone

def zone_report(tracking_data, point_name, zone_names, zone_label=None, invert=False):
    """
    Generates statistics for a body part in a zone.
    """
    if zone_label is None:
        if isinstance(zone_names, list):
            zone_label = ".".join(zone_names)
        else:
            zone_label = zone_names
            
    in_zone_mask = is_in_zone(tracking_data, point_name, zone_names, invert)
    if in_zone_mask is None:
        return {}
        
    df = tracking_data.data[point_name]
    fps = tracking_data.fps
    
    if 'is_moving' not in df.columns:
        is_moving = np.zeros(len(df), dtype=bool) 
    else:
        is_moving = df['is_moving'].fillna(False)
        
    raw_distance = df.loc[in_zone_mask, 'speed'].sum()
    distance_moving = df.loc[in_zone_mask & is_moving, 'speed'].sum()
    raw_speed = df.loc[in_zone_mask, 'speed'].mean() * fps if in_zone_mask.any() else 0
    speed_moving = df.loc[in_zone_mask & is_moving, 'speed'].mean() * fps if (in_zone_mask & is_moving).any() else 0
    time_moving = (in_zone_mask & is_moving).sum() / fps
    total_time = in_zone_mask.sum() / fps
    time_stationary = total_time - time_moving
    percentage_moving = (time_moving / total_time * 100) if total_time > 0 else 0
    
    padded = np.concatenate(([False], in_zone_mask))
    transitions = np.sum((padded[1:] == True) & (padded[:-1] == False))

    if in_zone_mask.any():
        first_frame = np.argmax(in_zone_mask)
        latency = first_frame / fps
    else:
        latency = np.nan
    
    report = {
        f"{zone_label}.raw.distance": raw_distance,
        f"{zone_label}.distance.moving": distance_moving,
        f"{zone_label}.raw.speed": raw_speed,
        f"{zone_label}.speed.moving": speed_moving,
        f"{zone_label}.time.moving": time_moving,
        f"{zone_label}.total.time": total_time,
        f"{zone_label}.time.stationary": time_stationary,
        f"{zone_label}.percentage.moving": percentage_moving,
        f"{zone_label}.transitions": transitions,
        f"{zone_label}.latency": latency
    }
    
    return report

def scale_polygon(polygon_df, factor):
    center_x = polygon_df['x'].mean()
    center_y = polygon_df['y'].mean()
    scaled = polygon_df.copy()
    scaled['x'] = center_x + factor * (polygon_df['x'] - center_x)
    scaled['y'] = center_y + factor * (polygon_df['y'] - center_y)
    return scaled

def recenter_polygon(polygon_df, new_center_df):
    center_x = polygon_df['x'].mean()
    center_y = polygon_df['y'].mean()
    if isinstance(new_center_df, pd.DataFrame):
        new_x = new_center_df['x'].values[0]
        new_y = new_center_df['y'].values[0]
    else:
        new_x = new_center_df['x']
        new_y = new_center_df['y']
    recentered = polygon_df.copy()
    recentered['x'] = polygon_df['x'] + new_x - center_x
    recentered['y'] = polygon_df['y'] + new_y - center_y
    return recentered

def add_oft_zones(tracking_data, corner_points, scale_center=0.5, scale_corners=0.4, scale_periphery=0.8):
    if not hasattr(tracking_data, 'median_data') or tracking_data.median_data is None:
        print("Median data needed for OFT zones")
        return tracking_data
        
    for p in corner_points:
        if p not in tracking_data.median_data.index:
            print(f"Point {p} not found in median data")
            return tracking_data
            
    if not hasattr(tracking_data, 'zones') or tracking_data.zones is None:
        tracking_data.zones = {}
        
    arena_poly = tracking_data.median_data.loc[corner_points, ['x', 'y']]
    tracking_data.zones['arena'] = arena_poly
    tracking_data.zones['center'] = scale_polygon(arena_poly, scale_center)
    tracking_data.zones['periphery'] = scale_polygon(arena_poly, scale_periphery)
    
    tracking_data.corner_names = []
    for p in corner_points:
        scaled = scale_polygon(arena_poly, scale_corners)
        corner_poly = recenter_polygon(scaled, tracking_data.median_data.loc[[p], ['x', 'y']])
        zone_name = f"corner.{p}"
        tracking_data.zones[zone_name] = corner_poly
        tracking_data.corner_names.append(zone_name)
        
    return tracking_data
