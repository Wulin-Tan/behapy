from typing import Optional, Union

import numpy as np
import umap as umap_lib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .._core._behapydata import BehapyData


def pca(
    bdata: BehapyData,
    n_comps: int = 50,
    zero_center: bool = True,
    svd_solver: str = "arpack",
    use_rep: Optional[str] = None,
    layer: Optional[str] = None,
) -> BehapyData:
    """
    Compute PCA coordinates.
    """
    if layer:
        X = bdata.layers[layer]
    elif use_rep:
        X = bdata.obsm[use_rep]
    else:
        X = bdata.X

    if zero_center:
        X = X - np.nanmean(X, axis=0)

    # Ensure n_comps is not greater than number of features/samples
    n_features = X.shape[1]
    n_samples = X.shape[0]
    if svd_solver == "arpack":
        n_comps = min(n_comps, n_features - 1, n_samples - 1)
    else:
        n_comps = min(n_comps, n_features, n_samples)
    
    # n_comps must be at least 1
    n_comps = max(1, n_comps)

    pca_obj = PCA(n_components=n_comps, svd_solver=svd_solver)
    X_pca = pca_obj.fit_transform(X)

    bdata.obsm["X_pca"] = X_pca
    bdata.varm["PCs"] = pca_obj.components_.T

    bdata.uns["pca"] = {
        "variance_ratio": pca_obj.explained_variance_ratio_,
        "variance": pca_obj.explained_variance_,
        "params": {
            "n_comps": n_comps,
            "zero_center": zero_center,
            "svd_solver": svd_solver,
        },
    }

    return bdata


def umap(
    bdata: BehapyData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    n_neighbors: Optional[int] = None,
    use_rep: str = "X_pca",
) -> BehapyData:
    """
    Compute UMAP embedding.
    """
    if use_rep == "X_pca" and "X_pca" not in bdata.obsm:
        pca(bdata)

    X = bdata.obsm[use_rep]

    if n_neighbors is None:
        if "neighbors" in bdata.uns:
            n_neighbors = bdata.uns["neighbors"]["n_neighbors"]
        else:
            n_neighbors = 15

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X)

    bdata.obsm["X_umap"] = X_umap
    bdata.uns["umap"] = {
        "params": {
            "min_dist": min_dist,
            "spread": spread,
            "n_components": n_components,
            "random_state": random_state,
            "n_neighbors": n_neighbors,
            "use_rep": use_rep,
        }
    }

    return bdata


def tsne(
    bdata: BehapyData,
    n_components: int = 2,
    perplexity: float = 30,
    early_exaggeration: float = 12,
    learning_rate: Union[float, str] = "auto",
    random_state: int = 0,
    use_rep: str = "X_pca",
) -> BehapyData:
    """
    Compute t-SNE embedding.
    """
    if use_rep == "X_pca" and "X_pca" not in bdata.obsm:
        pca(bdata)

    X = bdata.obsm[use_rep]

    tsne_obj = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    X_tsne = tsne_obj.fit_transform(X)

    bdata.obsm["X_tsne"] = X_tsne
    bdata.uns["tsne"] = {
        "params": {
            "n_components": n_components,
            "perplexity": perplexity,
            "early_exaggeration": early_exaggeration,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "use_rep": use_rep,
        }
    }

    return bdata
