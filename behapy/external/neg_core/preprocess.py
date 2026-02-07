import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def detect_center_shift(data, center_zone_radius=5, eps=2, min_pts=5):
    """
    Detect center zone points and calculate shift correction.
    """
    if 'InCentre' not in data.columns:
        raise ValueError("Data must contain 'InCentre' column")
        
    center_points = data[data['InCentre'] == 1][['Time', 'X', 'Y']].copy()
    n_points = len(center_points)
    
    if n_points == 0:
        return {
            'x_shift': 0, 'y_shift': 0, 'success': False,
            'reason': "no_points", 'n_points': 0, 'method': "none",
            'cluster_points': None, 'centroid': None, 'low_confidence': True
        }
    elif n_points < min_pts:
        center_points['cluster'] = 1
        centroid = center_points[['X', 'Y']].mean()
        return {
            'x_shift': -centroid['X'], 'y_shift': -centroid['Y'],
            'success': True, 'n_points': n_points, 'cluster_points': center_points,
            'centroid': centroid, 'method': "average", 'low_confidence': True
        }
        
    coords = center_points[['X', 'Y']].values
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(coords)
    labels = db.labels_
    
    center_points['cluster'] = labels
    
    if np.max(labels) == -1:
        center_points['cluster'] = 1
        centroid = center_points[['X', 'Y']].mean()
        return {
            'x_shift': -centroid['X'], 'y_shift': -centroid['Y'],
            'success': True, 'n_points': n_points, 'cluster_points': center_points,
            'centroid': centroid, 'method': "average", 'low_confidence': True
        }
        
    counts = center_points['cluster'].value_counts()
    valid_clusters = counts[counts.index != -1]
    
    if valid_clusters.empty:
        center_points['cluster'] = 1
        centroid = center_points[['X', 'Y']].mean()
        return {
            'x_shift': -centroid['X'], 'y_shift': -centroid['Y'],
            'success': True, 'n_points': n_points, 'cluster_points': center_points,
            'centroid': centroid, 'method': "average", 'low_confidence': True
        }

    main_cluster_label = valid_clusters.idxmax()
    cluster_points = center_points[center_points['cluster'] == main_cluster_label]
    centroid = cluster_points[['X', 'Y']].mean()
    
    return {
        'x_shift': -centroid['X'], 'y_shift': -centroid['Y'],
        'success': True, 'n_points': n_points, 'cluster_points': cluster_points,
        'centroid': centroid, 'method': "dbscan", 'low_confidence': False
    }

def apply_center_shift(data, shift_values):
    """
    Apply shift correction to coordinates.
    """
    df = data.copy()
    
    if not shift_values['success']:
        df['x_m'] = df['X']
        df['y_m'] = df['Y']
        return df
        
    df['x_m'] = df['X'] + shift_values['x_shift']
    df['y_m'] = df['Y'] + shift_values['y_shift']
    
    return df
