import pytest
import numpy as np
import pandas as pd
import behapy as bp
import os

@pytest.fixture
def mock_bdata():
    """Create a mock BehaviorData object with some clustering and temporal info."""
    # Synthetic data
    n_frames = 1000
    X = np.random.rand(n_frames, 10)
    obs = pd.DataFrame({
        'leiden': np.random.choice(['A', 'B', 'C'], n_frames),
        'speed': np.random.rand(n_frames)
    })
    bdata = bp.BehapyData(X=X, obs=obs)
    
    # Add transitions
    bp.tl.compute_transitions(bdata, key='leiden')
    # Add bouts
    bp.tl.detect_bouts(bdata, key='leiden', min_duration=5)
    bp.tl.compute_bout_statistics(bdata, key='leiden')
    
    return bdata

def test_compare_groups(mock_bdata):
    bdata2 = mock_bdata.copy()
    bdata2.obs['speed'] += 0.5 # Shift speed
    
    results = bp.tl.compare_groups(
        [mock_bdata, bdata2], 
        groups=['ctrl', 'treat'], 
        metric='speed', 
        test='t-test'
    )
    
    assert isinstance(results, pd.DataFrame)
    assert 'pvalue' in results.columns
    assert len(results) == 1

def test_behavior_frequency(mock_bdata):
    bdata2 = mock_bdata.copy()
    results = bp.tl.test_behavior_frequency(mock_bdata, bdata2)
    
    assert isinstance(results, pd.DataFrame)
    assert 'behavior' in results.columns
    assert 'pvalue' in results.columns

def test_bout_metrics(mock_bdata):
    bdata2 = mock_bdata.copy()
    results = bp.tl.test_bout_metrics(mock_bdata, bdata2, metric='duration')
    
    assert isinstance(results, pd.DataFrame)
    if not results.empty:
        assert 'cluster' in results.columns
        assert 'pvalue' in results.columns

def test_effect_size():
    d1 = np.random.normal(0, 1, 100)
    d2 = np.random.normal(1, 1, 100)
    es, (low, high) = bp.tl.compute_effect_size(d1, d2)
    
    assert isinstance(es, float)
    assert low < es < high

def test_bootstrap_ci():
    data = np.random.normal(0, 1, 100)
    low, high = bp.tl.bootstrap_ci(data, np.mean, n_bootstrap=100)
    
    assert low < np.mean(data) < high

def test_transition_matrix(mock_bdata):
    bdata2 = mock_bdata.copy()
    results = bp.tl.test_transition_matrix(mock_bdata, bdata2, test='permutation', n_permutations=10)
    
    assert isinstance(results, pd.DataFrame)
    assert 'from_behavior' in results.columns
    assert 'to_behavior' in results.columns
    assert 'diff' in results.columns
    assert 'pvalue' in results.columns
