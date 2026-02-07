# behapy Project Summary

## Overview
`behapy` is a Scanpy-like framework designed for analyzing high-dimensional behavioral time series data. It leverages the powerful `AnnData` structure to provide a modular, efficient, and user-friendly workflow for behavioral neuroscience, specifically tailored for pose estimation data from tools like DeepLabCut and SLEAP.

The project aims to standardize behavioral analysis by providing a comprehensive suite of tools for preprocessing, feature engineering, manifold learning, clustering, and temporal analysis, all while maintaining a familiar API for users of the single-cell genomics ecosystem.

## Quick Facts
- **Current version**: 0.3.0
- **License**: BSD-3-Clause
- **Python requirements**: >=3.8
- **Primary dependencies**: `anndata`, `scanpy`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `umap-learn`, `numba`, `h5py`, `tqdm`, `leidenalg`, `annoy`.

## Architecture

### Module Structure
- **`behapy/`**: Core package directory.
    - **`_core/`**: Contains `BehapyData`, the central data structure wrapping `AnnData`.
    - **`io/`**: Input/Output modules for reading DLC/SLEAP files and writing H5AD/CSV.
    - **`preprocessing/` (bp.pp)**: Functions for QC, filtering, smoothing, normalization, and feature extraction.
    - **`tools/` (bp.tl)**: Analytical tools including dimensionality reduction (PCA, UMAP, TSNE), clustering (Leiden, HDBSCAN), and temporal analysis.
    - **`external/` (bh.external)**: Integrated external tools (BehaviorFlow, NEG, VAME) for specialized behavioral analyses.
    - **`plotting/` (bp.pl)**: Visualization tools for trajectories, embeddings, behaviors, and temporal patterns.
    - **`datasets/`**: Utilities for generating synthetic behavioral data.
    - **`utils/`**: Internal utility functions and validation helpers.
- **`scripts/`**: Example workflows and validation scripts for testing the pipeline on real-world datasets (e.g., PyRAT).
- **`docs/`**: Documentation files, including validation results and dataset information.

### Data Model
`behapy` uses the `BehapyData` object, which inherits from `anndata.AnnData`. Data is organized as follows:
- **`.X`**: The primary data matrix (frames × features). Usually contains normalized kinematic features.
- **`.obs`**: Annotations for each frame (e.g., `speed`, `likelihood`, `leiden` clusters).
- **`.var`**: Annotations for each feature (e.g., feature names, types).
- **`.obsm`**: Multi-dimensional arrays for each frame (e.g., `X_pca`, `X_umap`, raw `coords`).
- **`.obsp`**: Pairwise relationships between frames (e.g., `distances`, `connectivities` from neighbor search).
- **`.uns`**: Unstructured metadata (e.g., transition matrices, cluster labels, plot parameters).

## Complete API Reference

### behapy.io
- `read(path, software, ...)`: Universal reader for behavioral data with auto-detection support.
- `read_dlc(path)`: Specialized reader for DeepLabCut CSV or H5 files.
- `read_sleap(path)`: Specialized reader for SLEAP H5 files.
- `detect_software(path)`: Identifies the pose estimation software used for a given file.
- `write_h5ad(bdata, path)`: Saves the `BehapyData` object to a compressed H5AD file.
- `write_csv(bdata, path)`: Exports observation data to a CSV file.

### behapy.preprocessing (bp.pp)
- `calculate_qc_metrics(bdata)`: Computes quality control metrics like mean likelihood and missing values.
- `filter_frames(bdata, ...)`: Removes low-quality frames based on likelihood or other criteria.
- `filter_bodyparts(bdata, ...)`: Removes bodyparts with insufficient data quality.
- `interpolate_missing(bdata)`: Fills missing coordinate values using linear or spline interpolation.
- `smooth(bdata, method, ...)`: Applies smoothing filters (Savgol, Gaussian, Median) to coordinates.
- `smooth_savgol(bdata)`, `smooth_gaussian(bdata)`, `smooth_median(bdata)`: Specialized smoothing functions.
- `compute_features(bdata, features)`: Unified interface for batch kinematic feature computation.
- `compute_speed(bdata)`: Calculates instantaneous speed for specified bodyparts.
- `compute_acceleration(bdata)`: Calculates frame-to-frame changes in speed.
- `compute_jerk(bdata)`: Calculates the rate of change of acceleration.
- `compute_angular_velocity(bdata)`: Calculates the turning rate (angular velocity).
- `compute_distances(bdata)`: Calculates all pairwise distances between bodyparts.
- `compute_angles(bdata)`: Calculates all triplets of angles between bodyparts.
- `compute_bodypart_distance(bdata, p1, p2)`: Calculates Euclidean distance between two specific bodyparts.
- `compute_bodypart_angle(bdata, p1, p2, p3)`: Calculates the angle formed by three specific bodyparts.
- `normalize_total(bdata)`: Normalizes feature values to a constant total.
- `scale(bdata)`: Scales features to unit variance and zero mean.
- `log_transform(bdata)`: Applies $log(1+x)$ transformation to features.
- `quantile_normalization(bdata)`: Performs quantile normalization across frames.
- `egocentric_alignment(bdata)`: Rotates and translates coordinates to an animal-centered frame.
- `pixel_to_real(bdata, conversion_factor)`: Converts pixel coordinates to real-world units (e.g., cm).
- `center_coordinates(bdata)`: Subtracts the mean or a reference point from coordinates.
- `neighbors(bdata, method, ...)`: Constructs a nearest neighbor graph (supports 'annoy' for speed).

### behapy.tools (bp.tl)
- `pca(bdata, n_comps)`: Performs Principal Component Analysis for dimensionality reduction.
- `umap(bdata, ...)`: Computes UMAP embedding for non-linear manifold learning.
- `tsne(bdata, ...)`: Computes t-SNE embedding.
- `leiden(bdata, resolution)`: Performs Leiden community detection for behavioral clustering.
- `louvain(bdata)`: Performs Louvain clustering.
- `hdbscan(bdata)`: Performs density-based clustering using HDBSCAN.
- `rank_features_groups(bdata, groupby)`: Identifies features that distinguish behavioral clusters.
- `merge_clusters(bdata, key, method)`: Merges fine-grained clusters using hierarchical or manual methods.
- `coarse_grain_clusters(bdata, target_n)`: Automatically merges clusters to a target number.
- `compute_transitions(bdata, key)`: Calculates Markovian transition probabilities between clusters.
- `compute_transition_entropy(bdata, key)`: Measures behavioral predictability via Shannon entropy.
- `detect_bouts(bdata, key, min_duration)`: Identifies continuous episodes of stable behavior.
- `compute_bout_statistics(bdata, key)`: Computes frequency, duration, and time metrics for behavioral bouts.

### behapy.external
- `behaviorflow.calculate_movement(adata, ...)`: Movement statistics using BehaviorFlow.
- `behaviorflow.zone_analysis(adata, ...)`: Zone occupancy and transition analysis.
- `neg.analyze_exploration(adata, ...)`: Grid-based exploration metrics for EPM/OFT.
- `vame.init_new_project(...)`: Access to VAME's project initialization and core tools.

### behapy.plotting (bp.pl)
- `pca(bdata)`, `umap(bdata)`, `tsne(bdata)`: Standard embedding plots colored by cluster or feature.
- `embedding(bdata, basis)`: Flexible embedding visualization for any basis in `.obsm`.
- `trajectory(bdata, bodypart)`: Plots the spatial path of a bodypart over time.
- `trajectory_heatmap(bdata)`: 2D occupancy heatmap of animal position.
- `ethogram(bdata)`: Traditional horizontal timeline of behavioral states.
- `ethogram_temporal(bdata)`: Optimized timeline visualization for large datasets.
- `behavior_pie(bdata)`: Pie chart of total time spent in each behavioral cluster.
- `bout_duration_distribution(bdata)`: Box/violin plots of behavioral bout durations.
- `transition_matrix(bdata)`: Heatmap of transition probabilities between clusters.
- `rank_features_groups(bdata)`: Dotplots or heatmaps of cluster-specific features.
- `feature_group_heatmap(bdata)`: Heatmap showing feature values across clusters.
- `feature_time_heatmap(bdata)`: Heatmap of features over time.
- `time_series(bdata, key)`: Interactive or static line plots of features over time.

## Version History

| Version | Date | Key Features |
| :--- | :--- | :--- |
| **v0.3.0** | 2026-01-02 | Temporal analysis (transitions, entropy, bouts), ethograms. |
| **v0.2.3** | 2026-01-02 | Annoy optimization (13x faster neighbors), auto-method selection. |
| **v0.2.2** | 2026-01-02 | Kinematic feature engineering suite (acceleration, angles, distances). |
| **v0.2.1** | 2026-01-02 | Cluster post-processing (merging, coarse-graining). |
| **v0.2.0** | 2026-01-02 | PyRAT validation, optimized plotting, bug fixes. |
| **v0.1.0** | 2026-01-02 | Initial release with core DLC/SLEAP support and pipeline. |

## Validated Datasets

### PyRAT Dataset
- **Scope**: 26 DeepLabCut files from rodent Open Field Tests (OFT).
- **Success Rate**: 96.2% (25/26 files) passed the full pipeline.
- **Data Quality**: 0.957 mean likelihood across all body parts.
- **Results**: Successfully identified consistent behavioral motifs (grooming, locomotion, rearing) across files.

## Performance Benchmarks

### Pipeline Speed (36k frames)
| Step | Avg Time (s) | Description |
| :--- | :--- | :--- |
| **Load** | 0.27s | Reading CSV into BehapyData |
| **Preproc** | 0.30s | QC, Smoothing, Speed calculation |
| **PCA** | <0.1s | Dimensionality reduction (30 PCs) |
| **Neighbors** | 2.15s | Annoy-based graph construction |
| **UMAP** | 33.06s | Manifold embedding |
| **Leiden** | 3.52s | Clustering (resolution=0.5) |
| **Total** | **~40s** | End-to-end pipeline |

## Example Workflow
```python
import behapy as bp

# 1. Load and QC
bdata = bp.io.read("data/raw/R1D1.csv", software='deeplabcut')
bp.pp.calculate_qc_metrics(bdata)
bp.pp.interpolate_missing(bdata)
bp.pp.smooth(bdata)

# 2. Feature Engineering
bp.pp.compute_features(bdata, features='all')

# 3. Dimensionality Reduction & Clustering
bp.tl.pca(bdata, n_comps=30)
bp.pp.neighbors(bdata, method='annoy')
bp.tl.umap(bdata)
bp.tl.leiden(bdata, resolution=0.5)

# 4. Temporal Analysis
bp.tl.compute_transitions(bdata)
bp.tl.detect_bouts(bdata, min_duration=5)

# 5. Visualization
bp.pl.umap(bdata, color='leiden')
bp.pl.transition_matrix(bdata)
bp.pl.ethogram_temporal(bdata, start=0, end=5000)
```

## Known Limitations
- **Multi-animal Support**: Current version is optimized for single-animal tracking. Multi-animal data (e.g., social interaction) is not yet fully supported in the preprocessing pipeline.
- **3D Pose Data**: Primary focus is on 2D coordinates; 3D support is experimental.
- **Video Overlay**: Direct video annotation with cluster labels is planned but not currently implemented.

## Development Roadmap
- [ ] Multi-animal interaction analysis.
- [ ] Real-time behavior classification interface.
- [ ] Integration with NeuroConv for NWB support.
- [ ] Automated cluster naming using LLMs or pre-trained classifiers.

## Files Reference
- **Main pipeline test**: `scripts/test_pyrat_pipeline.py`
- **Validation docs**: `docs/validation.md`
- **Example data**: `data/raw/dlc/pyrat/R1D1.csv`
- **Key scripts**:
    - `scripts/test_all_pyrat_files.py`: Bulk validation.
    - `scripts/test_neighbors_speed.py`: Performance benchmarking.
    - `scripts/test_temporal_analysis.py`: Sequence analysis demo.

## For New AI Agents
If you're an AI helping with this project:
1. Start by reading `CHANGELOG.md` and `docs/validation.md` to understand the current state and performance.
2. Run `scripts/test_pyrat_pipeline.py` to verify the local environment.
3. Use `BehapyData` as the primary object for all operations; it is fully compatible with Scanpy's ecosystem.
4. When adding new tools, follow the `behapy/tools/` structure and ensure they operate on `bdata` in-place by default.
