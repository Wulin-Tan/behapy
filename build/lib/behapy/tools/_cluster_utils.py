from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from .._core._behapydata import BehapyData

def merge_clusters(
    bdata: BehapyData, 
    key: str = 'leiden', 
    method: str = 'hierarchy', 
    n_clusters: Optional[int] = 10, 
    resolution: Optional[float] = None
) -> BehapyData:
    """
    Merge fine-grained clusters into coarser categories.

    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing the original clustering.
    method
        Merging method:
        - 'hierarchy': Use hierarchical clustering on cluster centroids.
        - 'resolution': Re-run Leiden with a lower resolution.
    n_clusters
        Number of target clusters for 'hierarchy' method.
    resolution
        Resolution for 'resolution' method.

    Returns
    -------
    BehapyData object with merged clusters in bdata.obs[f'{key}_merged'].
    """
    if key not in bdata.obs.columns:
        raise ValueError(f"Key '{key}' not found in bdata.obs.")

    if method == 'resolution':
        if resolution is None:
            raise ValueError("Resolution must be provided for method='resolution'.")
        from ._clustering import leiden
        leiden(bdata, resolution=resolution, key_added=f'{key}_merged')
    
    elif method == 'hierarchy':
        if n_clusters is None:
            raise ValueError("n_clusters must be provided for method='hierarchy'.")
        
        # Determine basis for centroids
        basis = 'X_pca' if 'X_pca' in bdata.obsm else ('X_umap' if 'X_umap' in bdata.obsm else None)
        if basis is None:
            raise ValueError("No embedding (PCA or UMAP) found in bdata.obsm for hierarchical merging.")

        # Compute centroids
        clusters = bdata.obs[key].unique()
        centroids = []
        cluster_order = []
        for cluster in clusters:
            mask = bdata.obs[key] == cluster
            centroids.append(bdata.obsm[basis][mask].mean(axis=0))
            cluster_order.append(cluster)
        
        centroids = np.array(centroids)
        
        # Hierarchical clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        new_labels = agg.fit_predict(centroids)
        
        # Map back to original observations
        mapping = dict(zip(cluster_order, new_labels))
        bdata.obs[f'{key}_merged'] = pd.Categorical(bdata.obs[key].map(mapping))
    
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'hierarchy' or 'resolution'.")

    return bdata

def coarse_grain_clusters(
    bdata: BehapyData, 
    key: str = 'leiden', 
    target_n: int = 8, 
    basis: str = 'X_pca'
) -> BehapyData:
    """
    Automatic hierarchical merging to achieve a target number of clusters.

    Parameters
    ----------
    bdata
        BehapyData object.
    key
        Key in bdata.obs containing the original clustering.
    target_n
        Target number of clusters.
    basis
        Basis in bdata.obsm to use for computing centroids (e.g., 'X_pca', 'X_umap').

    Returns
    -------
    BehapyData object with coarse-grained clusters in bdata.obs[f'{key}_coarse'].
    """
    if key not in bdata.obs.columns:
        raise ValueError(f"Key '{key}' not found in bdata.obs.")
    
    if basis not in bdata.obsm:
        raise ValueError(f"Basis '{basis}' not found in bdata.obsm.")

    # Compute centroids
    clusters = bdata.obs[key].unique()
    if len(clusters) <= target_n:
        bdata.obs[f'{key}_coarse'] = bdata.obs[key]
        return bdata

    centroids = []
    cluster_order = []
    for cluster in clusters:
        mask = bdata.obs[key] == cluster
        centroids.append(bdata.obsm[basis][mask].mean(axis=0))
        cluster_order.append(cluster)
    
    centroids = np.array(centroids)
    
    # Hierarchical clustering
    agg = AgglomerativeClustering(n_clusters=target_n)
    new_labels = agg.fit_predict(centroids)
    
    # Map back to original observations
    mapping = dict(zip(cluster_order, new_labels))
    bdata.obs[f'{key}_coarse'] = pd.Categorical(bdata.obs[key].map(mapping))
    
    return bdata
