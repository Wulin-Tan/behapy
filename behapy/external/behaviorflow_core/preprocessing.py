import pandas as pd
import numpy as np

def clean_tracking_data(tracking_data, likelihood_cutoff=0.95, max_delta=None):
    """
    Cleans up tracking data. Low likelihood points are interpolated.
    
    Args:
        tracking_data (TrackingData): The tracking data object.
        likelihood_cutoff (float): Points with likelihood below this will be interpolated.
        max_delta (float): Points that move more than max_delta in one frame will be interpolated.
        
    Returns:
        TrackingData: The cleaned tracking data object.
    """
    print(f"Interpolating points with likelihood < {likelihood_cutoff}")
    if max_delta:
        print(f"Interpolating points with a maximum delta of {max_delta} {tracking_data.distance_units} per frame")
        
    for point_name, df in tracking_data.data.items():
        mask = df['likelihood'] < likelihood_cutoff
        
        if max_delta:
            diff = df[['x', 'y']].diff()
            dist = np.sqrt(diff['x']**2 + diff['y']**2)
            jump_mask = dist > max_delta
            mask = mask | jump_mask

        df.loc[mask, ['x', 'y']] = np.nan
        df[['x', 'y']] = df[['x', 'y']].interpolate(method='linear', limit_direction='both')
        tracking_data.data[point_name] = df

    return tracking_data

def calibrate_tracking_data(tracking_data, method, in_metric=None, points=None, ratio=None, new_units="cm"):
    """
    Calibrates tracking data from pixel to metric space.
    
    Args:
        tracking_data (TrackingData): The tracking data object.
        method (str): 'distance', 'area', or 'ratio'.
        in_metric (float): The known metric distance or area.
        points (list): List of point names used for calibration.
        ratio (float): Direct conversion ratio (px_to_cm).
        new_units (str): Name of the new units (default 'cm').
        
    Returns:
        TrackingData: Calibrated object.
    """
    px_to_cm = 1.0
    
    if method == "ratio":
        if ratio is not None:
            px_to_cm = ratio
        else:
            print("Warning: method ratio needs a valid ratio.")
            return tracking_data
            
    elif method == "distance":
        if in_metric is not None and points is not None and len(points) == 2:
            p1 = tracking_data.median_data.loc[points[0]]
            p2 = tracking_data.median_data.loc[points[1]]
            dist_px = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
            px_to_cm = in_metric / dist_px
        else:
            print("Warning: method distance requires in_metric and exactly 2 points.")
            return tracking_data
            
    elif method == "area":
        print("Warning: Area calibration not fully implemented yet.")
        return tracking_data
    else:
        print(f"Warning: Invalid method {method}")
        return tracking_data
        
    tracking_data.px_to_cm = px_to_cm
    tracking_data.distance_units = new_units
    
    for point_name, df in tracking_data.data.items():
        df['x'] = df['x'] * px_to_cm
        df['y'] = df['y'] * px_to_cm
        tracking_data.data[point_name] = df
        
    tracking_data.median_data['x'] = tracking_data.median_data['x'] * px_to_cm
    tracking_data.median_data['y'] = tracking_data.median_data['y'] * px_to_cm
    
    return tracking_data
