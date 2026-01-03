from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
from .._core._behapydata import BehapyData

def compute_transitions(
    bdata: BehapyData, 
    key: str = "leiden", 
    normalize: bool = True, 
    copy: bool = False
) -> Optional[BehapyData]:
    """
    Count cluster-to-cluster transitions (frame i -> i+1).
    
    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing cluster labels.
    normalize
        If True, rows sum to 1 (probabilities).
    copy
        Whether to return a copy or modify in place.
        
    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()

    if key not in bdata.obs:
        raise ValueError(f"Key '{key}' not found in bdata.obs.")

    labels = bdata.obs[key].values
    unique_labels = sorted(bdata.obs[key].unique())
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    n_clusters = len(unique_labels)
    
    # Initialize transition matrix
    matrix = np.zeros((n_clusters, n_clusters))
    
    # Count transitions
    for i in range(len(labels) - 1):
        from_idx = label_to_idx[labels[i]]
        to_idx = label_to_idx[labels[i+1]]
        matrix[from_idx, to_idx] += 1
        
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        
    bdata.uns[f"{key}_transitions"] = matrix
    bdata.uns[f"{key}_transition_labels"] = unique_labels
    
    return bdata if copy else None

def compute_transition_entropy(
    bdata: BehapyData, 
    key: str = "leiden", 
    copy: bool = False
) -> Optional[BehapyData]:
    """
    Calculate Shannon entropy per cluster (high = unpredictable).
    
    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing cluster labels.
    copy
        Whether to return a copy or modify in place.
        
    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()
        
    trans_key = f"{key}_transitions"
    if trans_key not in bdata.uns:
        compute_transitions(bdata, key=key, normalize=True)
        
    matrix = bdata.uns[trans_key]
    
    # Shannon entropy: -sum(p * log2(p))
    # Filter out zero probabilities to avoid log(0)
    entropy = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        p = matrix[i, :]
        p = p[p > 0]
        if len(p) > 0:
            entropy[i] = -np.sum(p * np.log2(p))
            
    bdata.uns[f"{key}_transition_entropy"] = entropy
    
    return bdata if copy else None

def detect_bouts(
    bdata: BehapyData, 
    key: str = "leiden", 
    min_duration: int = 10, 
    copy: bool = False
) -> Optional[BehapyData]:
    """
    Find continuous episodes >= min_duration frames.
    
    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing cluster labels.
    min_duration
        Minimum number of frames for a bout to be included.
    copy
        Whether to return a copy or modify in place.
        
    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()
        
    if key not in bdata.obs:
        raise ValueError(f"Key '{key}' not found in bdata.obs.")
        
    labels = bdata.obs[key].values
    bouts = []
    
    if len(labels) == 0:
        bdata.uns[f"{key}_bouts"] = bouts
        return bdata if copy else None
        
    current_label = labels[0]
    start_idx = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            duration = i - start_idx
            if duration >= min_duration:
                bouts.append({
                    "cluster": current_label,
                    "start": start_idx,
                    "end": i,
                    "duration": duration
                })
            current_label = labels[i]
            start_idx = i
            
    # Handle the last bout
    duration = len(labels) - start_idx
    if duration >= min_duration:
        bouts.append({
            "cluster": current_label,
            "start": start_idx,
            "end": len(labels),
            "duration": duration
        })
        
    bdata.uns[f"{key}_bouts"] = bouts
    return bdata if copy else None

def compute_bout_statistics(
    bdata: BehapyData, 
    key: str = "leiden", 
    copy: bool = False
) -> Optional[BehapyData]:
    """
    Compute mean duration, count, total frames per cluster.
    
    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing cluster labels.
    copy
        Whether to return a copy or modify in place.
        
    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()
        
    bout_key = f"{key}_bouts"
    if bout_key not in bdata.uns:
        detect_bouts(bdata, key=key)
        
    bouts = bdata.uns[bout_key]
    if not bouts:
        bdata.uns[f"{key}_bout_stats"] = pd.DataFrame()
        return bdata if copy else None
        
    df_bouts = pd.DataFrame(bouts)
    
    stats = df_bouts.groupby("cluster")["duration"].agg(["mean", "count", "sum"])
    stats.columns = ["mean_duration", "bout_count", "total_frames"]
    
    # Ensure all clusters are present
    unique_labels = sorted(bdata.obs[key].unique())
    stats = stats.reindex(unique_labels).fillna(0)
    
    bdata.uns[f"{key}_bout_stats"] = stats
    return bdata if copy else None
