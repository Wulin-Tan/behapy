# Behapy Package Summary

This document provides a comprehensive summary of the `behapy` Python package, including its structure, modules, functions, classes, and dependencies.

## Table of Contents
- [Package Structure](#package-structure)
- [Dependencies](#dependencies)
- [Modules](#modules)
  - [behapy (Root)](#behapy-root)
  - [behapy._core](#behapy_core)
  - [behapy.datasets](#behapy_datasets)
  - [behapy.io](#behapy_io)
  - [behapy.plotting](#behapy_plotting)
  - [behapy.preprocessing](#behapy_preprocessing)
  - [behapy.tools](#behapy_tools)
  - [behapy.utils](#behapy_utils)

---

## Package Structure

The `behapy` package is organized into several subpackages, each focusing on a specific aspect of behavioral data analysis:

- `_core`: Core data structures (e.g., `BehapyData`).
- `datasets`: Built-in and synthetic datasets.
- `io`: Reading and writing data from various formats (DLC, SLEAP, H5AD, CSV).
- `plotting`: Visualization tools for behavior, embeddings, trajectories, and statistics.
- `preprocessing`: Data cleaning, normalization, smoothing, and feature engineering.
- `tools`: Analytical tools for clustering, dimension reduction, temporal analysis, and statistical testing.
- `utils`: Internal utility functions and validation logic.

---

## Dependencies

The package relies on the following key libraries (pinned for stability):
- **Data Handling**: `numpy`, `pandas`, `anndata`, `h5py`
- **Visualization**: `matplotlib`, `seaborn`
- **Computation**: `scipy`, `numba`, `scikit-learn`
- **Analysis**: `umap-learn==0.5.5`, `hdbscan`, `leidenalg==0.10.0`, `python-igraph`, `networkx`, `python-louvain`, `statsmodels`

---

## Modules

### behapy (Root)

#### `behapy.__init__`
- **File Path**: `behapy/__init__.py`
- **Description**: Package entry point. Exposes main submodules and key classes/functions.
- **Imports**: `datasets`, `get`, `io`, `neighbors`, `plotting` (as `pl`), `preprocessing` (as `pp`), `tools` (as `tl`), `BehapyData`, `settings`, `get_settings`, `set_figure_params`, `__version__`.

#### `behapy._settings`
- **File Path**: `behapy/_settings.py`
- **Description**: Global settings and configuration for the package.
- **Classes**:
  - `Settings`: Dataclass for global configuration (verbosity, figsize, dpi, cache_dir, etc.).
- **Functions**:
  - `get_settings() -> Settings`: Returns the current settings object.
  - `set_figure_params(dpi: int = 80, figsize: tuple = (8, 5), ...)`: Configures default matplotlib parameters.

#### `behapy._version`
- **File Path**: `behapy/_version.py`
- **Description**: Version information.
- **Variables**: `__version__` (current: 0.4.0).

---

### behapy._core

#### `behapy._core._behapydata`
- **File Path**: `behapy/_core/_behapydata.py`
- **Description**: Defines the core data structure `BehapyData`.
- **Classes**:
  - `BehapyData(AnnData)`: Wraps `AnnData` to provide a structured format for behavioral data.
    - **Methods**:
      - `__init__(X, obs, var, uns, obsm, varm, layers, ...)`: Initializes the object and performs validation.
      - `_validate()`: Basic data structure validation.
      - `copy() -> BehapyData`: Returns a copy of the object.
    - **Attributes (Properties)**:
      - `n_frames`: Number of frames (observations).
      - `n_features`: Number of features (variables).

---

### behapy.datasets

#### `behapy.datasets._synthetic`
- **File Path**: `behapy/datasets/_synthetic.py`
- **Description**: Utilities for generating synthetic behavioral data.
- **Functions**:
  - `synthetic_data(n_frames: int = 1000, n_bodyparts: int = 5) -> BehapyData`: Generates a `BehapyData` object with random coordinates and likelihoods.

---

### behapy.io

#### `behapy.io._dlc`
- **File Path**: `behapy/io/_dlc.py`
- **Description**: Readers for DeepLabCut (DLC) data.
- **Functions**:
  - `read_dlc_h5(filepath: Union[str, Path], animal: Optional[str] = None) -> Dict[str, Any]`: Reads DLC H5 files.
  - `read_dlc_csv(filepath: Union[str, Path], animal: Optional[str] = None) -> Dict[str, Any]`: Reads DLC CSV files.

#### `behapy.io._readers`
- **File Path**: `behapy/io/_readers.py`
- **Description**: Unified reading interface and software detection.
- **Functions**:
  - `detect_software(filepath: Union[str, Path]) -> str`: Auto-detects if a file is from DLC or SLEAP.
  - `read(filepath, software="auto", **kwargs)`: Main entry point. Supports 'deeplabcut' and 'sleap' software parameters.

#### `behapy.io._sleap`
- **File Path**: `behapy/io/_sleap.py`
- **Description**: Readers for SLEAP data.
- **Functions**:
  - `read_sleap_h5(filepath: Union[str, Path]) -> Dict[str, Any]`: Reads SLEAP H5 files and extracts tracking data.
  - `read_sleap(filepath: Union[str, Path], animal: Optional[str] = None, **kwargs) -> BehapyData`: Reads SLEAP data into `BehapyData`.

#### `behapy.io._writers`
- **File Path**: `behapy/io/_writers.py`
- **Description**: Data export utilities.
- **Functions**:
  - `write_h5ad(bdata: BehapyData, filepath: Union[str, Path])`: Saves `BehapyData` to H5AD.
  - `write_csv(bdata: BehapyData, filepath: Union[str, Path])`: Exports coordinates to CSV.

---

### behapy.plotting

#### `behapy.plotting._behavior`
- **File Path**: `behapy/plotting/_behavior.py`
- **Description**: Behavior-specific visualization with automated downsampling for large datasets.
- **Functions**:
  - `ethogram()`: Creates an ethogram plot.
  - `behavior_pie()`: Pie chart of behavior distribution.
  - `bout_distribution()`: Histogram of bout durations.
  - `time_series()`: Optimized 1D time series plotting with intelligent downsampling.
  - `feature_time_heatmap()`: Heatmap of features over time with downsampling.

#### `behapy.plotting._embedding`
- **File Path**: `behapy/plotting/_embedding.py`
- **Description**: Dimension reduction visualization with `max_points` downsampling for large datasets.
- **Functions**:
  - `embedding()`: Plots embedding coordinates with automated downsampling.
  - `umap()`, `tsne()`, `pca()`, `pca_variance_ratio()`: Convenience wrappers.

#### `behapy.plotting._features`
- **File Path**: `behapy/plotting/_features.py`
- **Description**: Feature analysis visualization.
- **Functions**:
  - `rank_features_groups()`: Plots top ranked features for groups.
  - `feature_group_heatmap()`: Heatmap of features grouped by behavior.

#### `behapy.plotting._temporal`
- **File Path**: `behapy/plotting/_temporal.py`
- **Description**: Temporal analysis visualization.
- **Functions**:
  - `transition_matrix()`: Plots behavioral transition matrices with optional significance markers.
  - `bout_duration_distribution()`: Comparison of bout duration distributions across behaviors.

#### `behapy.plotting._statistics`
- **File Path**: `behapy/plotting/_statistics.py`
- **Description**: Statistical visualization for group comparisons.
- **Functions**:
  - `effect_sizes()`: Forest or bar plots of effect sizes (Cohen's d, Hedges' g).
  - `statistical_summary()`: Multi-panel summary of statistical test results.

#### `behapy.plotting._trajectory`
- **File Path**: `behapy/plotting/_trajectory.py`
- **Description**: Spatial visualization with performance optimizations for long recordings.
- **Functions**:
  - `trajectory()`: Plots spatial trajectories with downsampling.
  - `trajectory_heatmap()`: Heatmap of spatial occupancy.

---

### behapy.preprocessing

#### `behapy.preprocessing._features`
- **File Path**: `behapy.preprocessing._features`
- **Description**: High-level behavioral feature extraction.
- **Functions**: `compute_distances()`, `compute_speed()`, `compute_acceleration()`, `compute_jerk()`, `compute_angles()`, `compute_angular_velocity()`.

#### `behapy.preprocessing._filter`
- **File Path**: `behapy/preprocessing/_filter.py`
- **Description**: Data cleaning and filtering.
- **Functions**: `filter_frames()`, `filter_bodyparts()`, `interpolate_missing()`.

#### `behapy.preprocessing._neighbors`
- **File Path**: `behapy/preprocessing/_neighbors.py`
- **Description**: Connectivity graph construction using UMAP or Annoy.

#### `behapy.preprocessing._normalize`
- **File Path**: `behapy/preprocessing/_normalize.py`
- **Description**: Normalization and scaling.
- **Functions**: `normalize_total()`, `scale()`, `log_transform()`, `quantile_normalization()`.

#### `behapy.preprocessing._qc`
- **File Path**: `behapy/preprocessing/_qc.py`
- **Description**: Quality control and outlier detection.

#### `behapy.preprocessing._smooth`
- **File Path**: `behapy/preprocessing/_smooth.py`
- **Description**: Temporal smoothing (Savitzky-Golay, Gaussian, Median).

#### `behapy.preprocessing._transform`
- **File Path**: `behapy/preprocessing/_transform.py`
- **Description**: Coordinate transformations (egocentric alignment, pixel-to-real).

---

### behapy.tools

#### `behapy.tools._clustering`
- **File Path**: `behapy/tools/_clustering.py`
- **Description**: Behavioral state clustering algorithms.
- **Functions**: `leiden()`, `louvain()`, `hdbscan()`, `kmeans()`, `hierarchical_clustering()`.

#### `behapy.tools._cluster_utils`
- **File Path**: `behapy/tools/_cluster_utils.py`
- **Description**: Utilities for refining clusters.
- **Functions**: `merge_clusters()`, `coarse_grain_clusters()`.

#### `behapy.tools._embedding`
- **File Path**: `behapy/tools/_embedding.py`
- **Description**: Dimension reduction techniques (PCA, UMAP, t-SNE).

#### `behapy.tools._temporal`
- **File Path**: `behapy/tools/_temporal.py`
- **Description**: Temporal analysis tools.
- **Functions**: `compute_transitions()`, `compute_transition_entropy()`, `detect_bouts()`, `compute_bout_statistics()`.

#### `behapy.tools._statistics`
- **File Path**: `behapy/tools/_statistics.py`
- **Description**: Statistical testing module for group comparisons (v0.4.0).
- **Functions**:
  - `compare_groups()`: Compares metrics across groups using permutation tests or standard tests (t-test, Mann-Whitney).
  - `test_transition_matrix()`: Element-wise comparison of transition matrices.
  - `test_behavior_frequency()`: Compares behavior occurrence frequencies.
  - `test_bout_metrics()`: Compares bout-level metrics (duration, frequency).
  - `compute_effect_size()`: Calculates Cohen's d or Hedges' g.
  - `bootstrap_ci()`: Computes confidence intervals via bootstrapping.

---

### behapy.utils

#### `behapy.utils._validation`
- **File Path**: `behapy/utils/_validation.py`
- **Description**: Internal data validation checks.

---

## Recent Improvements (v0.4.0-statistical-suite)

### Statistical Testing Module
- Added a comprehensive suite of statistical tests for comparing control vs. treatment groups.
- Supports permutation tests, t-tests, Mann-Whitney U tests, and Chi-square/Fisher's exact tests.
- Automated multiple comparison correction (FDR-BH, Bonferroni).
- Effect size computation and bootstrap confidence intervals.

### Temporal Analysis Enhancements
- Transition matrix computation with significance testing.
- Significance markers (stars) integrated into transition matrix visualizations.
- Entropy computation for behavioral transitions.

### Visualization Improvements
- New statistical plots: `effect_sizes()` (forest/bar) and `statistical_summary()`.
- Enhanced `transition_matrix()` with comparison modes and significance overlays.
- Standardized Seaborn integration for high-quality figures.

### Performance & Stability
- Continued use of intelligent downsampling for high-performance plotting on 1M+ frame datasets.
- Refined PCA component handling for small feature sets.
- Extensive unit test coverage for new statistical modules.
