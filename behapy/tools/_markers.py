from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._core._behapydata import BehapyData


def rank_features_groups(
    bdata: BehapyData,
    groupby: str,
    groups: Union[str, List[str]] = "all",
    reference: str = "rest",
    method: str = "wilcoxon",
    n_features: int = 100,
    use_rep: Optional[str] = None,
) -> BehapyData:
    """
    Rank features for characterizing groups.

    Parameters
    ----------
    bdata
        BehapyData object.
    groupby
        The key of the observations grouping to consider.
    groups
        Subset of groups, e.g. ['group1', 'group2'], to which comparison shall be restricted,
        or 'all' (default), for all groups.
    reference
        If 'rest', compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    method
        The statistical test for differential expression: 'wilcoxon' (Wilcoxon rank-sum),
        't-test', or 'logreg' (logistic regression).
    n_features
        The number of features that shall be returned as top-ranked.
    use_rep
        Which representation to use. If None, uses bdata.X.

    Returns
    -------
    BehapyData object with results in bdata.uns['rank_features_groups'].
    """
    if groupby not in bdata.obs.columns:
        raise ValueError(f"groupby {groupby} not found in bdata.obs.")

    if use_rep:
        X = bdata.obsm[use_rep]
    else:
        X = bdata.X

    all_groups = bdata.obs[groupby].unique()
    if groups == "all":
        groups = all_groups.tolist()

    n_vars = X.shape[1]
    n_features = min(n_features, n_vars)
    feature_names = bdata.var_names.tolist()

    # Storage for results
    names_array = np.zeros((n_features,), dtype=[(group, "U50") for group in groups])
    scores_array = np.zeros((n_features,), dtype=[(group, "f4") for group in groups])
    pvals_array = np.zeros((n_features,), dtype=[(group, "f4") for group in groups])

    for group in groups:
        # Get indices for group and reference
        group_mask = bdata.obs[groupby] == group
        if reference == "rest":
            ref_mask = ~group_mask
        else:
            ref_mask = bdata.obs[groupby] == reference

        X_group = X[group_mask]
        X_ref = X[ref_mask]

        scores = np.zeros(n_vars)
        pvals = np.ones(n_vars)

        if method == "wilcoxon":
            for i in range(n_vars):
                try:
                    res = stats.ranksums(X_group[:, i], X_ref[:, i])
                    scores[i] = res.statistic
                    pvals[i] = res.pvalue
                except:
                    scores[i] = 0
                    pvals[i] = 1
        elif method == "t-test":
            for i in range(n_vars):
                try:
                    res = stats.ttest_ind(X_group[:, i], X_ref[:, i], equal_var=False)
                    scores[i] = res.statistic
                    pvals[i] = res.pvalue
                except:
                    scores[i] = 0
                    pvals[i] = 1
        elif method == "logreg":
            from sklearn.linear_model import LogisticRegression

            # Binary classification
            y = np.zeros(X.shape[0])
            y[group_mask] = 1
            # Combine group and ref for training
            train_mask = group_mask | ref_mask
            X_train = X[train_mask]
            y_train = y[train_mask]

            clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            scores = clf.coef_[0]
            pvals = np.ones(n_vars)  # Logreg doesn't give standard p-values easily
        else:
            raise ValueError(f"Unknown method {method}")

        # Rank features
        idx = np.argsort(np.abs(scores))[::-1][:n_features]

        names_array[group] = [feature_names[i] for i in idx]
        scores_array[group] = scores[idx]
        pvals_array[group] = pvals[idx]

    bdata.uns["rank_features_groups"] = {
        "params": {
            "groupby": groupby,
            "reference": reference,
            "method": method,
            "use_rep": use_rep,
        },
        "names": names_array,
        "scores": scores_array,
        "pvals": pvals_array,
    }

    return bdata
