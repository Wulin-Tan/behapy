# PyRAT Dataset Validation Results

This document summarizes the validation of `behapy` on the PyRAT dataset, which consists of DeepLabCut tracking data from rodent behavior experiments.

## Dataset Overview
- **Total Files**: 26
- **Average Frames**: ~30,000 per file (up to 37,335)
- **Data Quality**: 0.957 mean likelihood across all body parts.

## Validation Strategy
A two-tier validation strategy was used to ensure both broad compatibility and deep functional correctness.

### Tier 1: Quick Validation (26 files)
**Status**: 25/26 Success (96.2%)
- **Pipeline**: Load → QC → Smooth → Speed
- **Performance**: ~0.34s per file (0.01s per 1k frames).
- **Failures**: `PlexonTracking.csv` failed due to incompatible column headers (missing 'x', 'y' labels), which is expected for this specific format.

### Tier 2: Full Pipeline (7 selected files)
**Status**: 7/7 Success (100%)
- **Pipeline**: Load → QC → Smooth → Speed → PCA → Neighbors → UMAP → Leiden
- **Performance**: Average 60.78s per file (for ~36k frames).
- **Clustering**: Produced 77-157 clusters at resolution=0.5.

## Performance Breakdown
| Step | Avg Time (s) | Description |
| :--- | :--- | :--- |
| **Load** | 0.27s | Reading CSV into BehapyData |
| **Preproc** | 0.30s | QC, Smoothing, Speed calculation |
| **PCA** | <0.1s | Dimensionality reduction |
| **Neighbors** | 2s | Nearest neighbor graph construction (Annoy) |
| **UMAP** | 20-39s | Manifold embedding |
| **Leiden** | 3-16s | Community detection (clustering) |

## Performance Optimization (v0.2.3)

In v0.2.3, we replaced the default UMAP-based neighbor search with **Annoy** (Approximate Nearest Neighbors Oh Yeah), achieving a significant speedup in the pipeline.

### Benchmark Results (36k frames, 30 PCs)
| Method | Neighbors Time | UMAP Time | Total Time | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Default (Scanpy) | 28.79s | 35.23s | 64.02s | 100% |
| **Annoy (v0.2.3)** | **2.15s** | **33.06s** | **35.21s** | **99.5%** |
| Exact (Sklearn) | 1.59s | 35.23s | 36.82s | 100% |

**Key Benefits:**
- **13x Speedup** in neighbor computation compared to the previous default.
- **1.8x Speedup** in the total clustering pipeline (Neighbors + UMAP).
- **99.5% Accuracy** compared to exact nearest neighbor search.
- **Scalability**: Annoy scales $O(N \log N)$, making it much faster for large datasets (>100k frames) where exact search becomes prohibitive.

### Usage
By default, `bp.pp.neighbors()` now uses `method='auto'`, which selects Annoy for datasets with >10,000 frames.

```python
# Automatic selection (Annoy if N > 10k)
bp.pp.neighbors(bdata, method='auto')

# Explicitly use Annoy
bp.pp.neighbors(bdata, method='annoy', n_trees=20)
```

## Example Usage
```python
import behapy as bp

# 1. Load data
bdata = bp.io.read("path/to/tracking.csv", software='deeplabcut')

# 2. Preprocessing
bp.pp.calculate_qc_metrics(bdata)
bp.pp.smooth(bdata)
bp.pp.compute_speed(bdata)

# 3. Analysis
bp.tl.pca(bdata)
bp.tl.neighbors(bdata)
bp.tl.umap(bdata)
bp.tl.leiden(bdata, resolution=0.5)

# 4. Visualization
bp.pl.trajectory(bdata, bodypart='nose')
bp.pl.umap(bdata, color='leiden')
bp.pl.time_series(bdata, key='speed')
```

### Kinematic Feature Engineering

behapy provides rich kinematic features beyond basic position tracking:

```python
import behapy as bp

bdata = bp.io.read('data.csv', software='deeplabcut')

# Individual features
bp.pp.compute_speed(bdata)
bp.pp.compute_acceleration(bdata)
bp.pp.compute_angular_velocity(bdata, bodypart='nose')

# Posture features
bp.pp.compute_bodypart_distance(bdata, 'nose', 'tail_base')
bp.pp.compute_bodypart_angle(bdata, 'nose', 'body', 'tail_base')

# Compute all standard features at once
bp.pp.compute_features(bdata, features='all')
```

Available features stored in `bdata.obs`:
- `speed`: Instantaneous velocity magnitude
- `acceleration`: Rate of speed change
- `angular_velocity`: Turning rate (radians/frame)
- `distance_X_Y`: Euclidean distance between bodyparts
- `angle_X_Y_Z`: Angle formed by three bodyparts

## Cluster Post-Processing

For users who need coarser behavioral categories, `behapy` provides two convenience functions:

```python
# Option 1: Hierarchical merging to target number
bp.tl.coarse_grain_clusters(bdata, key='leiden', target_n=8)
bp.pl.embedding(bdata, basis='umap', color='leiden_coarse')

# Option 2: Manual merging with hierarchy or re-run leiden
bp.tl.merge_clusters(bdata, key='leiden', method='hierarchy', n_clusters=10)
bp.pl.embedding(bdata, basis='umap', color='leiden_merged')

# Option 3: Simply re-run with lower resolution
bp.tl.leiden(bdata, resolution=0.1)  # Produces ~8-15 clusters
```

## Temporal Analysis (v0.3.0)

`behapy` now includes tools for analyzing the temporal structure of behavior, including transition probabilities, ethograms, and bout statistics.

### Key Metrics
- **Transitions**: Markovian transition probabilities between behavioral clusters.
- **Entropy**: Shannon entropy of transitions per cluster, indicating behavioral predictability.
- **Bouts**: Continuous episodes of the same behavior, with statistics on duration and frequency.

### Example Usage
```python
import behapy as bp

# 1. Compute temporal metrics
bp.tl.compute_transitions(bdata, key='leiden')
bp.tl.compute_transition_entropy(bdata, key='leiden')
bp.tl.detect_bouts(bdata, key='leiden', min_duration=5)
bp.tl.compute_bout_statistics(bdata, key='leiden')

# 2. Visualization
# Transition probability heatmap
bp.pl.transition_matrix(bdata, key='leiden')

# Horizontal timeline of behaviors
bp.pl.ethogram_temporal(bdata, key='leiden', start=0, end=5000)

# Distribution of bout durations per cluster
bp.pl.bout_duration_distribution(bdata, key='leiden')
```

### Validation
Validated on the R1D1 PyRAT dataset:
- **Mean Transition Entropy**: 0.077 (highly predictable transitions at frame-to-frame level).
- **Bout Detection**: Identified 274 stable behavioral bouts (>5 frames).
- **Visualization**: Successfully generated 3-panel temporal analysis summary.
```

## Validation Artifacts
- **Logs**: `logs/all_pyrat_test.log`
- **Summaries**: `logs/pyrat_tier1_summary.csv`, `logs/pyrat_tier2_summary.csv`
- **Visualizations**: `data/processed/pyrat_validation/`
