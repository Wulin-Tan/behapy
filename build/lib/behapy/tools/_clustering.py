from typing import Optional, Union

try:
    import community as louvain_lib
except ImportError:
    louvain_lib = None

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None

try:
    import leidenalg
except ImportError:
    leidenalg = None

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

from .._core._behapydata import BehapyData


def leiden(
    bdata: BehapyData,
    resolution: float = 1.0,
    random_state: int = 0,
    n_iterations: int = -1,
    key_added: str = "leiden",
    use_rep: Optional[str] = None,
) -> BehapyData:
    """
    Perform Leiden clustering on the neighbor graph.
    """
    if leidenalg is None:
        raise ImportError("Please install 'leidenalg' to use this function.")

    if "connectivities" not in bdata.obsp:
        raise ValueError("Neighbor graph not found. Run pp.neighbors() first.")

    import igraph as ig

    # Convert sparse matrix to igraph
    sources, targets = bdata.obsp["connectivities"].nonzero()
    weights = bdata.obsp["connectivities"].data
    g = ig.Graph(n=bdata.n_obs, edges=list(zip(sources, targets)), directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights="weight",
        n_iterations=n_iterations,
        seed=random_state,
    )

    bdata.obs[key_added] = pd.Categorical(partition.membership)
    bdata.uns["leiden"] = {
        "params": {
            "resolution": resolution,
            "random_state": random_state,
            "n_iterations": n_iterations,
        }
    }

    return bdata


def louvain(
    bdata: BehapyData, resolution: float = 1.0, random_state: int = 0, key_added: str = "louvain"
) -> BehapyData:
    """
    Perform Louvain clustering on the neighbor graph.
    """
    if louvain_lib is None:
        raise ImportError("Please install 'python-louvain' to use this function.")

    if "connectivities" not in bdata.obsp:
        raise ValueError("Neighbor graph not found. Run pp.neighbors() first.")

    import networkx as nx

    # Convert sparse matrix to networkx
    g = nx.from_scipy_sparse_array(bdata.obsp["connectivities"])

    partition = louvain_lib.best_partition(g, resolution=resolution, random_state=random_state)

    # Map back to nodes
    membership = [partition[i] for i in range(bdata.n_obs)]

    bdata.obs[key_added] = pd.Categorical(membership)
    bdata.uns["louvain"] = {"params": {"resolution": resolution, "random_state": random_state}}

    return bdata


def hdbscan(
    bdata: BehapyData,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    use_rep: str = "X_pca",
) -> BehapyData:
    """
    Perform HDBSCAN clustering.
    """
    if hdbscan_lib is None:
        raise ImportError("Please install 'hdbscan' to use this function.")

    if use_rep not in bdata.obsm:
        raise ValueError(f"Representation {use_rep} not found in bdata.obsm.")

    X = bdata.obsm[use_rep]

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = clusterer.fit_predict(X)

    bdata.obs["hdbscan"] = pd.Categorical(labels)
    bdata.obs["hdbscan_probabilities"] = clusterer.probabilities_
    bdata.uns["hdbscan"] = {
        "params": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "use_rep": use_rep,
        }
    }

    return bdata


def hierarchical_clustering(
    bdata: BehapyData,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    linkage: str = "ward",
    use_rep: str = "X_pca",
) -> BehapyData:
    """
    Perform Agglomerative (Hierarchical) clustering.
    """
    if use_rep not in bdata.obsm:
        raise ValueError(f"Representation {use_rep} not found in bdata.obsm.")

    X = bdata.obsm[use_rep]

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, distance_threshold=distance_threshold, linkage=linkage
    )
    labels = clusterer.fit_predict(X)

    bdata.obs["hierarchical"] = pd.Categorical(labels)
    bdata.uns["hierarchical"] = {
        "params": {
            "n_clusters": n_clusters,
            "distance_threshold": distance_threshold,
            "linkage": linkage,
            "use_rep": use_rep,
        }
    }

    return bdata


def kmeans(
    bdata: BehapyData,
    n_clusters: int = 8,
    random_state: int = 0,
    use_rep: str = "X_pca",
    key_added: str = "kmeans",
) -> BehapyData:
    """
    Perform K-Means clustering.

    Parameters
    ----------
    bdata
        BehapyData object.
    n_clusters
        The number of clusters to form.
    random_state
        Determines random number generation for centroid initialization.
    use_rep
        Which representation to use.
    key_added
        The key in bdata.obs where the labels will be stored.

    Returns
    -------
    BehapyData object with cluster labels in bdata.obs[key_added].
    """
    if use_rep not in bdata.obsm:
        raise ValueError(f"Representation {use_rep} not found in bdata.obsm.")

    X = bdata.obsm[use_rep]

    kmeans_obj = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans_obj.fit_predict(X)

    bdata.obs[key_added] = pd.Categorical(labels)
    bdata.uns["kmeans"] = {
        "params": {
            "n_clusters": n_clusters,
            "random_state": random_state,
            "use_rep": use_rep,
        }
    }

    return bdata
