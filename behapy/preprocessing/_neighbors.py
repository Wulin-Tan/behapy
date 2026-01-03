from typing import Optional, Union

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .._core._behapydata import BehapyData
from ._neighbors_annoy import compute_neighbors_annoy


def neighbors(
    bdata: BehapyData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    metric: str = "euclidean",
    method: str = "auto",
    **kwargs,
) -> BehapyData:
    """
    Compute k-nearest neighbors graph.

    Parameters
    ----------
    bdata
        BehapyData object.
    n_neighbors
        Number of neighbors to find.
    n_pcs
        Number of PCs to use. If provided, PCA is run first.
    use_rep
        Key in bdata.obsm to use.
    metric
        Distance metric.
    method
        Method for neighbor search: 'auto', 'exact', or 'annoy'.
        - 'auto': Use annoy if n_obs > 10000, else exact.
        - 'exact': Use scanpy/sklearn exact search.
        - 'annoy': Use approximate search with Annoy.
    **kwargs
        Additional arguments passed to specific implementations.
    """
    if method == "annoy" or (method == "auto" and bdata.n_obs > 10000):
        # Ensure PCA is run if n_pcs is requested
        if n_pcs and "X_pca" not in bdata.obsm:
            from ..tools._embedding import pca

            pca(bdata, n_comps=n_pcs)
            use_rep = "X_pca"

        return compute_neighbors_annoy(
            bdata, n_neighbors=n_neighbors, metric=metric, use_rep=use_rep or "X_pca", **kwargs
        )

    # Original scanpy/exact implementation
    if use_rep:
        X = bdata.obsm[use_rep]
    else:
        X = bdata.X

    if n_pcs:
        pca_obj = PCA(n_components=n_pcs)
        X = pca_obj.fit_transform(X)
        bdata.obsm["X_pca"] = X

    if method == "exact":
        # Use sklearn for exact
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        from scipy.sparse import csr_matrix

        n_obs = X.shape[0]
        rows = np.repeat(np.arange(n_obs), n_neighbors)
        cols = indices.flatten()
        data = np.ones(len(rows))
        connectivities = csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))

        bdata.obsp["connectivities"] = connectivities
        bdata.obsp["distances"] = csr_matrix((distances.flatten(), (rows, cols)), shape=(n_obs, n_obs))
    else:
        # Default to scanpy (which is also UMAP-based or exact)
        # We use scanpy here to ensure compatibility with bdata (AnnData-like)
        sc.pp.neighbors(bdata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep, metric=metric)

    return bdata
