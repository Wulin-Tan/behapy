import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
from tqdm import tqdm
from typing import List, Union, Callable, Optional, Tuple, Dict
import warnings

from .._core._behapydata import BehapyData

def compare_groups(
    bdata_list: List[BehapyData],
    groups: List[str],
    metric: str,
    test: str = 'permutation',
    n_permutations: int = 10000,
    correction: str = 'fdr_bh'
) -> pd.DataFrame:
    """
    Compare behavioral metrics between multiple groups.

    Parameters
    ----------
    bdata_list
        List of BehapyData objects.
    groups
        List of group labels corresponding to each bdata in bdata_list.
    metric
        Name of the metric to compare (must exist in bdata.obs, bdata.uns, or be a calculated property).
        Supported metrics: 'transition_frequency', 'bout_duration', 'bout_count', 'total_time'.
    test
        Statistical test to use ('permutation', 't-test', 'mannwhitney', 'chi-square').
    n_permutations
        Number of permutations for the permutation test.
    correction
        Multiple comparison correction method ('fdr_bh', 'bonferroni', 'holm').

    Returns
    -------
    pd.DataFrame
        Test results with [comparison, statistic, pvalue, pvalue_corrected, effect_size].
    """
    unique_groups = sorted(list(set(groups)))
    if len(unique_groups) < 2:
        raise ValueError("At least two groups are required for comparison.")

    results = []
    
    # Helper to extract metric values per group
    def get_values(group_label):
        indices = [i for i, g in enumerate(groups) if g == group_label]
        values = []
        for idx in indices:
            bdata = bdata_list[idx]
            if metric == 'bout_duration':
                if 'leiden_bout_stats' in bdata.uns:
                    values.extend(bdata.uns['leiden_bout_stats']['mean_duration'].values)
            elif metric == 'bout_count':
                if 'leiden_bout_stats' in bdata.uns:
                    values.extend(bdata.uns['leiden_bout_stats']['count'].values)
            elif metric == 'total_time':
                if 'leiden_bout_stats' in bdata.uns:
                    values.extend(bdata.uns['leiden_bout_stats']['total_frames'].values)
            elif metric in bdata.obs:
                values.extend(bdata.obs[metric].dropna().values)
            else:
                # Fallback or specific extraction logic
                pass
        return np.array(values)

    # Perform pairwise comparisons
    import itertools
    for g1, g2 in itertools.combinations(unique_groups, 2):
        data1 = get_values(g1)
        data2 = get_values(g2)
        
        if len(data1) == 0 or len(data2) == 0:
            warnings.warn(f"Empty data for groups {g1} or {g2}. Skipping.")
            continue
            
        if len(data1) < 10 or len(data2) < 10:
            warnings.warn(f"Small sample size for {g1} (n={len(data1)}) or {g2} (n={len(data2)}).")

        stat, pval = 0.0, 1.0
        if test == 't-test':
            stat, pval = stats.ttest_ind(data1, data2, equal_var=False)
        elif test == 'mannwhitney':
            stat, pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        elif test == 'permutation':
            # Basic permutation test for difference in means
            observed_diff = np.mean(data1) - np.mean(data2)
            combined = np.concatenate([data1, data2])
            count = 0
            for _ in tqdm(range(n_permutations), desc=f"Permuting {g1} vs {g2}", leave=False):
                shuffled = np.random.permutation(combined)
                new_diff = np.mean(shuffled[:len(data1)]) - np.mean(shuffled[len(data1):])
                if abs(new_diff) >= abs(observed_diff):
                    count += 1
            stat = observed_diff
            pval = count / n_permutations
        
        eff_size = compute_effect_size(data1, data2, method='cohen_d')[0]
        
        results.append({
            'comparison': f"{g1} vs {g2}",
            'statistic': stat,
            'pvalue': pval,
            'effect_size': eff_size
        })

    df = pd.DataFrame(results)
    if not df.empty and correction:
        _, p_adj, _, _ = multipletests(df['pvalue'], method=correction)
        df['pvalue_corrected'] = p_adj
    
    return df

def test_transition_matrix(
    bdata1: BehapyData,
    bdata2: BehapyData,
    key: str = 'leiden',
    test: str = 'permutation',
    n_permutations: int = 10000
) -> pd.DataFrame:
    """
    Compare transition matrices element-wise between two BehapyData objects.

    Returns
    -------
    pd.DataFrame with [from_behavior, to_behavior, diff, pvalue, significant].
    """
    trans_key = f"{key}_transitions"
    if trans_key not in bdata1.uns or trans_key not in bdata2.uns:
        raise ValueError(f"Transition matrix '{trans_key}' not found in uns.")

    m1 = bdata1.uns[trans_key]
    m2 = bdata2.uns[trans_key]
    labels = bdata1.uns.get(f"{key}_transition_labels", list(range(m1.shape[0])))
    
    if m1.shape != m2.shape:
        raise ValueError("Transition matrices must have the same shape.")

    diff = m1 - m2
    pvalues = np.ones_like(m1)
    
    if test == 'permutation':
        # Permutation test for transition matrices
        # We shuffle the group labels of the transitions
        def get_transitions(bdata, k):
            labels = bdata.obs[k].values
            return list(zip(labels[:-1], labels[1:]))
        
        t1 = get_transitions(bdata1, key)
        t2 = get_transitions(bdata2, key)
        
        combined_t = t1 + t2
        n1 = len(t1)
        
        # Pre-calculate labels mapping
        unique_labels = sorted(list(set([t[0] for t in combined_t] + [t[1] for t in combined_t])))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        n_labels = len(unique_labels)
        
        def compute_matrix(transitions):
            m = np.zeros((n_labels, n_labels))
            for f, t in transitions:
                m[label_to_idx[f], label_to_idx[t]] += 1
            # Normalize rows
            row_sums = m.sum(axis=1, keepdims=True)
            return np.divide(m, row_sums, out=np.zeros_like(m), where=row_sums!=0)

        obs_m1 = compute_matrix(t1)
        obs_m2 = compute_matrix(t2)
        observed_diff = obs_m1 - obs_m2
        
        counts = np.zeros_like(observed_diff)
        
        for _ in tqdm(range(n_permutations), desc="Permuting transitions"):
            indices = np.random.permutation(len(combined_t))
            p_t1 = [combined_t[i] for i in indices[:n1]]
            p_t2 = [combined_t[i] for i in indices[n1:]]
            
            p_m1 = compute_matrix(p_t1)
            p_m2 = compute_matrix(p_t2)
            p_diff = p_m1 - p_m2
            
            counts += (np.abs(p_diff) >= np.abs(observed_diff))
            
        pvalues = counts / n_permutations
        diff = observed_diff
        labels = unique_labels

    results = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            pval = pvalues[i, j] if test == 'permutation' else 1.0
            results.append({
                'from_behavior': labels[i],
                'to_behavior': labels[j],
                'diff': diff[i, j],
                'pvalue': pval,
                'significant': pval < 0.05
            })
            
    return pd.DataFrame(results)

def test_behavior_frequency(
    bdata1: BehapyData,
    bdata2: BehapyData,
    key: str = 'leiden',
    behaviors: Optional[List[str]] = None,
    test: str = 'chi2'
) -> pd.DataFrame:
    """Compare behavior occurrence frequencies between two datasets."""
    freq1 = bdata1.obs[key].value_counts(normalize=False)
    freq2 = bdata2.obs[key].value_counts(normalize=False)
    
    all_behaviors = behaviors if behaviors else sorted(list(set(freq1.index) | set(freq2.index)))
    
    results = []
    for b in all_behaviors:
        c1 = freq1.get(b, 0)
        c2 = freq2.get(b, 0)
        n1 = len(bdata1.obs) - c1
        n2 = len(bdata2.obs) - c2
        
        contingency = [[c1, n1], [c2, n2]]
        
        if test == 'chi2':
            stat, pval, _, _ = stats.chi2_contingency(contingency)
        elif test == 'fisher':
            stat, pval = stats.fisher_exact(contingency)
        else:
            stat, pval = 0, 1.0
            
        results.append({
            'behavior': b,
            'count1': c1,
            'count2': c2,
            'statistic': stat,
            'pvalue': pval
        })
        
    return pd.DataFrame(results)

def test_bout_metrics(
    bdata1: BehapyData,
    bdata2: BehapyData,
    key: str = 'leiden',
    metric: str = 'duration',
    test: str = 'mannwhitney'
) -> pd.DataFrame:
    """Compare bout durations, counts, or latencies."""
    bouts1 = bdata1.uns.get(f"{key}_bouts", [])
    bouts2 = bdata2.uns.get(f"{key}_bouts", [])
    
    if not bouts1 or not bouts2:
        raise ValueError(f"Bouts not found for key '{key}'. Run detect_bouts first.")
        
    df1 = pd.DataFrame(bouts1)
    df2 = pd.DataFrame(bouts2)
    
    unique_clusters = sorted(list(set(df1['cluster']) | set(df2['cluster'])))
    results = []
    
    for cluster in unique_clusters:
        val1 = df1[df1['cluster'] == cluster][metric].values
        val2 = df2[df2['cluster'] == cluster][metric].values
        
        if len(val1) < 2 or len(val2) < 2:
            continue
            
        if test == 'mannwhitney':
            stat, pval = stats.mannwhitneyu(val1, val2)
        elif test == 't-test':
            stat, pval = stats.ttest_ind(val1, val2, equal_var=False)
        else:
            stat, pval = 0, 1.0
            
        results.append({
            'cluster': cluster,
            'statistic': stat,
            'pvalue': pval,
            'n1': len(val1),
            'n2': len(val2)
        })
        
    return pd.DataFrame(results)

def compute_effect_size(
    group1_data: np.ndarray,
    group2_data: np.ndarray,
    method: str = 'cohen_d',
    ci: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """Compute effect size and confidence interval."""
    n1, n2 = len(group1_data), len(group2_data)
    m1, m2 = np.mean(group1_data), np.mean(group2_data)
    v1, v2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
    
    if method == 'cohen_d':
        pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        es = (m1 - m2) / pooled_std
    elif method == 'hedges_g':
        pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        cohen_d = (m1 - m2) / pooled_std
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        es = cohen_d * correction
    else:
        es = 0.0
        
    # Bootstrap CI for effect size
    def es_func(d1, d2):
        nm1, nm2 = np.mean(d1), np.mean(d2)
        nv1, nv2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        npooled = np.sqrt(((len(d1)-1)*nv1 + (len(d2)-1)*nv2) / (len(d1)+len(d2)-2))
        return (nm1 - nm2) / npooled if npooled != 0 else 0
        
    ci_low, ci_high = bootstrap_ci((group1_data, group2_data), es_func, n_bootstrap=1000)
    
    return es, (ci_low, ci_high)

def bootstrap_ci(
    data: Union[np.ndarray, Tuple[np.ndarray, ...]],
    metric_func: Callable,
    n_bootstrap: int = 10000,
    ci: float = 0.95
) -> Tuple[float, float]:
    """Generic bootstrap confidence interval."""
    boot_stats = []
    for _ in range(n_bootstrap):
        if isinstance(data, tuple):
            samples = tuple(resample(d) for d in data)
            boot_stats.append(metric_func(*samples))
        else:
            sample = resample(data)
            boot_stats.append(metric_func(sample))
            
    alpha = 1.0 - ci
    lower = np.percentile(boot_stats, alpha / 2 * 100)
    upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)
    return lower, upper
