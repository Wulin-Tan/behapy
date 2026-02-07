import time
from typing import Optional

import numpy as np
from annoy import AnnoyIndex
from scipy.sparse import csr_matrix

from .._core._behapydata import BehapyData


def compute_neighbors_annoy(
    bdata: BehapyData,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    n_trees: int = 10,
    search_k: int = -1,
    use_rep: str = "X_pca",
    copy: bool = False,
) -> Optional[BehapyData]:
    """
    Compute approximate k-nearest neighbors using Annoy.

    Parameters
    ----------
    bdata
        BehapyData object.
    n_neighbors
        Number of neighbors to find.
    metric
        Distance metric: 'euclidean' or 'angular'.
    n_trees
        Number of trees for Annoy index. More trees = better accuracy.
    search_k
        Search effort. -1 means auto (n_trees * n_neighbors).
    use_rep
        Key in bdata.obsm to use for neighbor search.
    copy
        Whether to return a copy or modify in place.

    Returns
    -------
    BehapyData object if copy=True, else None.
    """
    if copy:
        bdata = bdata.copy()

    if use_rep not in bdata.obsm:
        raise ValueError(f"Representation '{use_rep}' not found in bdata.obsm.")

    X = bdata.obsm[use_rep]
    n_obs, n_features = X.shape

    print(f"Building Annoy index with {n_trees} trees...")
    start_time = time.time()

    # Build Annoy index
    # Annoy uses 'angular' for cosine distance, 'euclidean' for L2
    annoy_metric = "angular" if metric == "cosine" else metric
    t = AnnoyIndex(n_features, annoy_metric)
    for i in range(n_obs):
        t.add_item(i, X[i])
    t.build(n_trees)

    index_time = time.time() - start_time
    print(f"Annoy index built in {index_time:.2f}s")

    print(f"Searching for {n_neighbors} neighbors...")
    start_search = time.time()

    indices = np.zeros((n_obs, n_neighbors), dtype=int)
    distances = np.zeros((n_obs, n_neighbors), dtype=float)

    for i in range(n_obs):
        idx, dist = t.get_nns_by_item(i, n_neighbors, search_k=search_k, include_distances=True)
        indices[i, :] = idx
        distances[i, :] = dist

    search_time = time.time() - start_search
    print(f"Neighbor search completed in {search_time:.2f}s")

    # Convert to sparse matrices (matching scanpy format)
    rows = np.repeat(np.arange(n_obs), n_neighbors)
    cols = indices.flatten()
    dists = distances.flatten()

    # Create sparse distance matrix
    bdata.obsp["distances"] = csr_matrix((dists, (rows, cols)), shape=(n_obs, n_obs))

    # Create sparse connectivity matrix (binary for now, or weights)
    conns = np.ones(len(rows))
    bdata.obsp["connectivities"] = csr_matrix((conns, (rows, cols)), shape=(n_obs, n_obs))

    # Store metadata
    bdata.uns["neighbors"] = {
        "n_neighbors": n_neighbors,
        "metric": metric,
        "method": "annoy",
        "use_rep": use_rep,
        "params": {"n_trees": n_trees, "search_k": search_k},
    }

    return bdata if copy else None
