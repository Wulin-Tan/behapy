---
title: Reference Projects Summary
description: Comprehensive overview of Python reference projects including A-SOID, PyRAT, Scanpy, DLC2Kinematics, ehrapy, and scaleSC
date: 2025-12-26
---
# Reference Projects Summary

This document provides a comprehensive overview of key Python reference projects relevant to behavioral analysis, single-cell genomics, and motion capture analysis. Each section details the directory structure, core functionality, key classes/functions, and technology stack.

## Table of Contents

- [A-SOID (Active-SOiD)](#a-soid-active-soid)
- [PyRAT](#pyrat)
- [Scanpy](#scanpy)
- [DLC2Kinematics](#dlc2kinematics)
- [ehrapy](#ehrapy)
- [scaleSC](#scalesc)

---

## A-SOID (Active-SOiD)

### Overview

A-SOID (Active-SOiD) is a no-code web application for building supervised classifiers for animal behavior analysis using pose estimation data. The codebase is organized into a Streamlit-based GUI with a modular workflow spanning data preprocessing, feature extraction, active learning, manual refinement, prediction, and unsupervised discovery.

### Directory Structure

#### Entry Points

- **`app.py`** – Main Streamlit application. Defines the GUI layout, navigation, and integration of all app modules.
  - Functions: `img_to_bytes`, `img_to_html`, `index`, `main`
- **`__main__.py`** – CLI entry point using Click. Provides command `app` to launch the Streamlit app with configuration.
  - Functions: `load_streamlit_config`, `conv_config_to_args`, `main` (CLI group), `main_streamlit`

#### Apps Modules (Step-by-Step Workflow)

Each module corresponds to a step in the A-SOiD pipeline and contains a `main()` function that is called from the sidebar navigation.

1. **`apps/A_data_preprocess.py`** – Project creation and data upload.

   - `main(config=None)` – Handles project configuration and data preprocessing.
2. **`apps/B_extract_features.py`** – Feature extraction from pose data.

   - `prompt_setup(...)` – Interactive parameter setup for feature extraction.
   - `main(config=None)` – Manages feature extraction workflow.
3. **`apps/C_auto_active_learning.py`** – Active learning for classifier training.

   - `prompt_setup(...)` – Sets active-learning parameters (initial ratio, max iterations, etc.).
   - `main(ri=None, config=None)` – Orchestrates the active-learning process.
4. **`apps/D_manual_active_learning.py`** – Manual refinement of low-confidence predictions.

   - `main(ri=None, config=None)` – Launches the refinement interface.
5. **`apps/E_create_new_training.py`** – Incorporates refined data into a new training set.

   - `create_new_training_features_targets(...)` – Merges new features/targets with existing data.
   - `main(ri=None, config=None)` – Creates a new iteration dataset.
6. **`apps/F_predict.py`** – Predict behaviors on new pose data and generate visualizations.

   - Numerous helper functions for plotting (pie charts, ethograms, bout durations), video annotation, and label smoothing.
   - `main(ri=None, config=None)` – Prediction and visualization pipeline.
7. **`apps/G_unsupervised_discovery.py`** – Unsupervised clustering to discover behavioral subtypes.

   - `prompt_setup(...)`, `get_features_labels(...)`, `hdbscan_classification(...)`, `pca_umap_hdbscan(...)`, `plot_hdbscan_embedding(...)`, `save_update_info(...)`
   - `main(ri=None, config=None)` – Discovers and splits behaviors into subtypes.

#### Utils Modules (Core Functionality)

##### Feature Extraction

- **`utils/extract_features.py`** – High-level feature extraction class and interactive duration distribution plot.
  - `interactive_durations_dist(...)` – Plots behavior bout durations.
  - `class Extract` – Orchestrates 2D/3D feature extraction, label down-sampling, and saving.
- **`utils/extract_features_2D.py`** – Numba-accelerated 2D feature extraction routines.
  - `fast_standardize`, `fast_running_mean`, `fast_feature_extraction`, `fast_feature_binning`, `feature_extraction`, etc.
- **`utils/extract_features_3D.py`** – Numba-accelerated 3D feature extraction routines.
  - Similar functions adapted for 3D data.

##### Machine Learning / Active Learning

- **`utils/auto_active_learning.py`** – Implements the active-learning loop with a random-forest classifier.
  - `show_classifier_results(...)` – Visualizes performance across iterations.
  - `class RF_Classify` – Manages subsampled classification, self-learning, and model saving.
- **`utils/predict.py`** – Prediction functions for both 2D and 3D data.
  - `bsoid_predict_numba`, `bsoid_predict_numba_noscale`, `frameshift_predict`, etc.

##### Data Loading and Workspace Management

- **`utils/load_workspace.py`** – Numerous load/save utilities for project data (features, models, refinements, etc.).
  - `load_data`, `load_features`, `load_iterX`, `load_refinement`, `save_data`, etc.
- **`utils/load_preprocess.py`** – Data preprocessing and software selection.
  - `convert_data_format`, `select_software`
- **`utils/import_data.py`** – Loaders for various pose-estimation formats (DLC, SLEAP, OpenMonkeyStudio) and annotation files (BORIS).
  - `load_pose`, `load_labels`, `get_bodyparts`, etc.
- **`utils/preprocessing.py`** – Filtering and sorting utilities.
  - `adp_filt`, `sort_nicely`, `get_filenames`, etc.

##### Project Utilities

- **`utils/project_utils.py`** – Configuration management and project creation.
  - `load_default_config`, `load_config`, `update_config`, `create_new_project`, `view_config_md`
- **`utils/manual_refinement.py`** – Video frame extraction and labeled-video creation for manual refinement.
  - `frame_extraction`, `create_labeled_vid`, `prompt_setup`
- **`utils/motionenergy.py`** – Egocentric alignment and motion-energy calculations.
  - `conv_2_egocentric`, `animate_blobs`, `calc_motion_energy_single`
- **`utils/load_css.py`** – CSS loading for Streamlit.

#### Config Modules

- **`config/global_config.py`** – UMAP and HDBSCAN default parameters.
- **`config/help_messages.py`** – Extensive help strings used throughout the GUI.
- **`config/default_config.ini`** – Default configuration template.

### Key Classes

1. **`Extract`** (`utils/extract_features.py`) – Manages feature extraction pipeline.
2. **`RF_Classify`** (`utils/auto_active_learning.py`) – Handles active-learning classification.
3. **`Refinement`** (`utils/manual_refinement.py`) – Not shown in the snippets but used in `D_manual_active_learning.py`.
4. **`Preprocess`** (`utils/load_preprocess.py`) – Used in `A_data_preprocess.py`.

### Function Count

Approximately **170 function definitions** across the codebase, excluding class methods.

### Workflow Summary

The pipeline follows a linear, iterative workflow:

1. **Data Preprocessing** – Upload pose and annotation files, create project.
2. **Feature Extraction** – Compute spatiotemporal features from pose data.
3. **Active Learning** – Train a classifier with a small labeled set, iteratively add low-confidence samples.
4. **Manual Refinement** (optional) – Manually correct low-confidence predictions.
5. **Create New Dataset** – Merge refined data into training set.
6. **Predict** – Apply trained classifier to new pose data, generate ethograms, videos, and statistics.
7. **Unsupervised Discovery** (optional) – Cluster behaviors into subtypes using UMAP + HDBSCAN.

### Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn (Random Forest), UMAP, HDBSCAN
- **Acceleration**: Numba (for feature extraction)
- **Data Formats**: DLC CSV, SLEAP H5, BORIS binary, OpenMonkeyStudio MAT/TXT
- **Visualization**: Plotly, OpenCV (video annotation), Matplotlib

---

## PyRAT

### Overview

The `pyrat/pyratlib` directory contains two Python files:

- `__init__.py`: Imports core dependencies and exposes all functions from `processing.py`.
- `processing.py`: Contains 23 functions for processing and analyzing animal tracking data (primarily from DeepLabCut), electrophysiology data, and behavior classification.

### Dependencies

The library relies on:

- `numpy`, `matplotlib.pyplot`, `pandas`, `csv`, `matplotlib.cm`
- Additional imports within functions: `mpl_toolkits.axes_grid1`, `matplotlib.patches`, `neo.io.BlackrockIO`, `sklearn.manifold.TSNE`, `sklearn.cluster.AgglomerativeClustering`, `cv2`, `scipy.cluster.hierarchy.dendrogram`, etc.

### Functions in `processing.py`

| Function                        | Description                                                                       | Key Parameters                                                                                                                              | Returns                                                                         |
| ------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `Trajectory`                  | Plots trajectory of a body part over time.                                        | `data` (DataFrame), `bodyPart` (str), `bodyPartBox` (str), `**kwargs` (start, end, fps, cmapType, etc.)                             | Plot (matplotlib figure)                                                        |
| `Heatmap`                     | Plots heatmap of body part trajectory density.                                    | `data`, `bodyPart`, `**kwargs` (similar to Trajectory)                                                                                | Plot                                                                            |
| `pixel2centimeters`           | Converts pixel values to real-world scale.                                        | `data`, `pixel_max`, `pixel_min`, `max_real`, `min_real`                                                                          | Scale factor (float)                                                            |
| `MotionMetrics`               | Computes velocity, acceleration, and distance traveled.                           | `data`, `bodyPart`, `filter`, `fps`, `max_real`, `min_real`                                                                     | DataFrame with metrics                                                          |
| `FieldDetermination`          | Creates a DataFrame defining rectangular/circular regions of interest.            | `Fields`, `plot`, `**kwargs` (data, bodyPartBox, posit, obj_color)                                                                    | DataFrame of field coordinates (optionally plot)                                |
| `Interaction`                 | Quantifies interaction between a body part and defined fields.                    | `data`, `bodyPart`, `fields`, `fps`                                                                                                 | DataFrame of interactions, list of raw interactions                             |
| `Reports`                     | Generates a summary report for multiple datasets.                                 | `df_list`, `list_name`, `bodypart`, `fields`, `filter`, `fps`                                                                   | DataFrame with aggregated metrics                                               |
| `DrawLine`                    | Draws an arrow representing orientation (used by `HeadOrientation`).            | `x`, `y`, `angle`, `**kwargs` (arrow_width, head_width, etc.)                                                                       | Matplotlib arrow object                                                         |
| `HeadOrientation`             | Plots trajectory with arrows indicating head direction.                           | `data`, `step`, `head`, `tail`, `**kwargs` (start, end, fps, etc.)                                                                | Plot                                                                            |
| `SignalSubset`                | Extracts subsets of electrophysiology data based on time markers.                 | `sig_data`, `freq`, `fields`, `**kwargs` (start_time, end_time)                                                                     | Dictionary of subsets per channel/object                                        |
| `LFP`                         | Extracts LFP channels from a MATLAB .mat file (Plexon-converted).                 | `data` (.mat file)                                                                                                                        | DataFrame with LFP data                                                         |
| `PlotInteraction`             | Plots a horizontal bar showing interaction times with fields.                     | `interactions`, `**kwargs` (start, end, fps, figureTitle, etc.)                                                                         | Plot                                                                            |
| `Blackrock`                   | Converts a Blackrock .ns2 file to a DataFrame of LFP channels.                    | `data_path`, `freq`                                                                                                                     | DataFrame with LFP data                                                         |
| `SpacialNeuralActivity`       | Creates a heatmap of neural activity in pixel space.                              | `neural_data`, `unit`                                                                                                                   | 2D array (heatmap)                                                              |
| `IntervalBehaviors`           | Extracts time intervals of classified behaviors.                                  | `cluster_labels`, `fps`, `filter`, `correction`                                                                                     | Dictionary of intervals per cluster                                             |
| `TrajectoryMA`                | Plots trajectory for multi-animal data (DLC .h5 format).                          | `data`, `bodyPart`, `bodyPartBox`, `**kwargs` (animals, joint_plot, etc.)                                                           | Plot(s)                                                                         |
| `splitMultiAnimal`            | Splits multi-animal tracking data into per-animal dictionaries.                   | `data`, `data_type`, `**kwargs` (animals, bodyParts, start, end, fps)                                                                 | Dictionary of animals → body parts → coordinates                              |
| `multi2single`                | Extracts a single animal from multi-animal data to a standard DataFrame.          | `data`, `animal`, `data_type`, `**kwargs` (animals, bodyParts, start, end, fps, drop)                                               | DataFrame compatible with other PyRAT functions                                 |
| `distance_metrics`            | Computes pairwise distances between body parts per frame.                         | `data`, `bodyparts_list`, `distance`                                                                                                  | Array of distances (high-dimensional)                                           |
| `model_distance`              | Creates t-SNE embedding and hierarchical clustering models.                       | `dimensions`, `distance`, `n_jobs`, `verbose`, `perplexity`, `learning_rate`                                                    | `model` (AgglomerativeClustering), `embedding` (TSNE)                       |
| `ClassifyBehaviorMultiVideos` | Classifies behavior across multiple videos using distance metrics and clustering. | `data` (dict of DataFrames), `bodyparts_list`, `dimensions`, `distance`, `**kwargs` (n_jobs, verbose, etc.)                       | `cluster_df`, `cluster_coord`, `fitted_model`                             |
| `dendrogram`                  | Plots a dendrogram for a hierarchical clustering model.                           | `model`, `**kwargs`                                                                                                                     | Matplotlib dendrogram                                                           |
| `ClassifyBehavior`            | Classifies behavior for a single video; saves frame images per cluster.           | `data`, `video`, `bodyparts_list`, `dimensions`, `distance`, `**kwargs` (startIndex, endIndex, directory, return_metrics, etc.) | `cluster_labels`, `X_transformed`, `model`, `d` (plus optional metrics) |

### Notes

- Functions are designed to work with DeepLabCut (DLC) output formats (CSV or HDF5).
- Many functions accept extensive `**kwargs` for customization of plotting parameters, time ranges, and filtering.
- The library supports integration with electrophysiology data (Blackrock, Plexon) and behavior classification via t-SNE + hierarchical clustering.
- Multi-animal tracking is supported through `TrajectoryMA`, `splitMultiAnimal`, and `multi2single`.

### Usage

Import the library as:

```python
import pyratlib as rat
```

All functions are available under the `rat` namespace (e.g., `rat.Trajectory(...)`).

---

## Scanpy

### Overview

The `scanpy/src/scanpy` directory contains the core Python modules for the Scanpy library, organized into submodules for preprocessing, tools, plotting, datasets, neighbors, metrics, queries, and more. Below is a comprehensive list of modules and their exported functions/classes.

### Module Breakdown

#### 1. Top-Level Modules (directly under `scanpy/src/scanpy/`)

- `__init__.py` – exports the main API: `pp`, `tl`, `pl`, `datasets`, `neighbors`, `read`, `write`, `settings`, etc.
- `readwrite.py` – functions for reading/writing data:
  - `read`, `read_10x_h5`, `read_10x_mtx`, `read_visium` (deprecated), `write`
  - `read_params`, `write_params`
  - helper functions: `is_valid_filename`, `_download`, etc.
- `logging.py` – logging utilities:
  - `print_header`, `print_version_and_date`, `print_versions` (deprecated)
  - `error`, `warning`, `info`, `hint`, `debug`
  - `print_memory_usage`, `get_memory_usage`
- `cli.py` – command-line interface: `main`, `console_main`
- `_compat.py`, `_singleton.py`, `_types.py` – internal support modules.

#### 2. Preprocessing (`preprocessing/`)

Exported functions (as `scanpy.pp`):

- **Normalization & scaling**: `normalize_total`, `normalize_per_cell`, `scale`, `log1p`, `sqrt`
- **Filtering**: `filter_cells`, `filter_genes`, `filter_genes_dispersion`
- **Quality control**: `calculate_qc_metrics`
- **Batch correction**: `combat`
- **Feature selection**: `highly_variable_genes`
- **Dimensionality reduction**: `pca`
- **Doublet detection**: `scrublet`, `scrublet_simulate_doublets`
- **Downsampling**: `downsample_counts`, `sample`, `subsample`
- **Regression**: `regress_out`
- **Recipes**: `recipe_seurat`, `recipe_weinreb17`, `recipe_zheng17`
- **Neighborhood graph**: `neighbors` (re-exported from `neighbors` module)

#### 3. Tools (`tools/`)

Exported functions (as `scanpy.tl`):

- **Clustering**: `leiden`, `louvain`
- **Embedding**: `umap`, `tsne`, `diffmap`, `draw_graph`
- **Trajectory inference**: `dpt` (diffusion pseudotime)
- **Graph-based tools**: `paga`, `ingest`
- **Marker gene detection**: `rank_genes_groups`, `filter_rank_genes_groups`
- **Gene scoring**: `score_genes`, `score_genes_cell_cycle`
- **Visualization aids**: `dendrogram`, `embedding_density`, `marker_gene_overlap`
- **Simulation**: `sim`
- **PCA**: `pca` (accessed via `__getattr__`)

#### 4. Plotting (`plotting/`)

Exported functions and classes (as `scanpy.pl`):

- **Scatter plots**: `scatter`, `embedding`, `pca`, `tsne`, `umap`, `diffmap`, `draw_graph`, `spatial`
- **Heatmaps**: `heatmap`, `clustermap`, `matrixplot`, `dotplot`, `stacked_violin`
- **Ranking plots**: `ranking`, `rank_genes_groups`, `rank_genes_groups_dotplot`, `rank_genes_groups_heatmap`, `rank_genes_groups_matrixplot`, `rank_genes_groups_stacked_violin`, `rank_genes_groups_tracksplot`, `rank_genes_groups_violin`
- **Trajectory plots**: `dpt_groups_pseudotime`, `dpt_timeseries`
- **Quality control**: `highest_expr_genes`, `filter_genes_dispersion`, `highly_variable_genes`, `scrublet_score_distribution`
- **PCA-specific**: `pca_loadings`, `pca_overview`, `pca_scatter`, `pca_variance_ratio`
- **PAGA plots**: `paga`, `paga_compare`, `paga_path`
- **Utilities**: `matrix`, `set_rcParams_defaults`, `set_rcParams_scanpy`, `palettes`
- **Classes**: `DotPlot`, `MatrixPlot`, `StackedViolin`
- **Easter egg**: `dogplot`

#### 5. Datasets (`datasets/`)

Exported dataset loaders (as `scanpy.datasets`):

- `blobs`, `burczynski06`, `ebi_expression_atlas`, `krumsiek11`, `moignard15`, `paul15`, `pbmc3k`, `pbmc3k_processed`, `pbmc68k_reduced`, `toggleswitch`, `visium_sge`

#### 6. Neighbors (`neighbors/`)

Primary exports:

- `neighbors` – main function to compute k-nearest-neighbor graph.
- `Neighbors` – class encapsulating neighborhood graphs and related computations.
- Supporting classes: `FlatTree`, `OnFlySymMatrix`.

#### 7. External (`external/`)

Submodules that wrap external tools:

- `pp` – external preprocessing methods.
- `tl` – external analysis tools.
- `pl` – external plotting functions.
- `exporting` – export utilities.

#### 8. Get (`get/`)

Functions to extract data from AnnData objects:

- `obs_df`, `var_df`, `rank_genes_groups_df`, `aggregate`
- Internal helpers: `_get_obs_rep`, `_set_obs_rep`, `_check_mask`, `_ObsRep`

#### 9. Metrics (`metrics/`)

Spatial autocorrelation and evaluation metrics:

- `morans_i`, `gearys_c`, `confusion_matrix`

#### 10. Queries (`queries/`)

Biomart and enrichment queries:

- `biomart_annotations`, `gene_coordinates`, `mitochondrial_genes`, `enrich` (g:Profiler)

#### 11. Simulation Models (`sim_models/`)

Package for simulating single-cell RNA-seq data (no top-level functions exported).

#### 12. Experimental (`experimental/`)

Experimental features; currently only contains `pp` submodule.

#### 13. Settings (`_settings/`)

Not directly imported; provides `settings` object and `Verbosity` enum.

#### 14. Utilities (`_utils/`)

Internal utilities for type annotation, documentation, etc.

### Summary Statistics

- **Total submodules**: 14 (excluding internal `_utils` and `_settings`).
- **Exported functions**: ~80+ (excluding classes and internal functions).
- **Primary namespaces**: `pp`, `tl`, `pl`, `datasets`, `neighbors`, `get`, `metrics`, `queries`, `external`.
- **Key classes**: `Neighbors`, `DotPlot`, `MatrixPlot`, `StackedViolin`.

---

## DLC2Kinematics

### Overview

DLC2Kinematics is a Python library for kinematic analysis of DeepLabCut (DLC) motion capture data. It provides tools for loading, smoothing, computing velocities, joint angles, quaternions, synergies, dimensionality reduction (PCA, UMAP), and visualization.

### File-by-File Summary

#### 1. `__init__.py`

- **Purpose**: Module exports and version definition.
- **Key Exports**:
  - `__version__`, `VERSION` from `version.py`
  - `load_data`, `smooth_trajectory` from `preprocess.py`
  - `load_c3d_data` from `preprocess_c3d.py`
  - `compute_velocity`, `compute_acceleration`, `compute_speed`, `extract_kinematic_synergies`, `compute_umap` from `mainfxns.py`
  - `plot_joint_angles`, `plot_velocity`, `pca_plot`, `plot_3d_pca_reconstruction`, `visualize_synergies`, `plot_umap` from `plotting.py`
  - `load_joint_angles`, `compute_joint_angles`, `compute_joint_velocity`, `compute_joint_acceleration`, `compute_correlation`, `compute_pca` from `joint_analysis.py`
  - `compute_joint_quaternions`, `compute_joint_doubleangles`, `plot_joint_quaternions`, `compute_joint_quaternion_velocity`, `compute_joint_quaternion_acceleration`, `_load_quaternions` from `quaternions.py`
  - `Visualizer3D`, `MinimalVisualizer3D`, `MultiVisualizer`, `Visualizer2D` from `visualization.py`
  - `auxiliaryfunctions` from `utils`

#### 2. `preprocess.py`

- **Purpose**: Loading and smoothing of DLC data (HDF5 format).
- **Key Functions**:
  - `load_data(filename, smooth=False, filter_window=3, order=1)`: Loads DLC multi-index pandas array, optionally applies Savitzky-Golay smoothing.
  - `smooth_trajectory(df, bodyparts, filter_window=3, order=1, deriv=0, save=False, ...)`: Smooths selected bodyparts, can compute derivatives (velocity/acceleration).

#### 3. `preprocess_c3d.py`

- **Purpose**: Import and preprocess 3D motion capture data from C3D files.
- **Key Functions**:
  - `load_c3d_data(filename, scorer="scorer", smooth=False, filter_window=3, order=1)`: Reads C3D file, reshapes to DLC-like DataFrame.
  - `get_data_from_c3d_file(filename)`: Extracts raw data, bodypart labels, frame range, and sample rate.
  - `create_empty_df(scorer, bodyparts, frames_no)`: Creates empty DataFrame with proper multi-index structure.
  - `get_c3d_bodyparts(handle)`: Reads bodypart labels from C3D file.

#### 4. `mainfxns.py`

- **Purpose**: Core kinematic computations.
- **Key Functions**:
  - `compute_velocity(df, bodyparts, filter_window=3, order=1)`: Computes velocity via Savitzky-Golay derivative.
  - `compute_acceleration(df, bodyparts, filter_window=3, order=2)`: Computes acceleration.
  - `compute_speed(df, bodyparts, filter_window=3, order=1)`: Computes speed (norm of velocity vectors).
  - `extract_kinematic_synergies(data, tol=0.95, num_syn=None, standardize=False, ampl=1)`: Performs PCA-based synergy extraction.
  - `compute_umap(df, keypoints=None, pcutoff=0.6, chunk_length=30, fit_transform=True, ...)`: Computes UMAP embedding on pose data.

#### 5. `joint_analysis.py`

- **Purpose**: Joint-angle-based analysis.
- **Key Functions**:
  - `load_joint_angles(data)`: Loads previously saved joint angles.
  - `compute_joint_angles(df, joints_dict, save=True, destfolder=None, ...)`: Computes joint angles from 3-point segments.
  - `compute_joint_velocity(joint_angle, filter_window=3, order=1, save=True, ...)`: Computes angular velocity.
  - `compute_joint_acceleration(joint_angle, filter_window=3, order=2, save=True, ...)`: Computes angular acceleration.
  - `compute_correlation(feature, plot=False, colormap="viridis")`: Computes correlation matrix of joint features.
  - `compute_pca(feature, n_components=None, plot=True, alphaValue=0.7)`: Performs PCA on joint features.

#### 6. `quaternions.py`

- **Purpose**: Quaternion-based joint rotation representation.
- **Key Functions**:
  - `compute_joint_quaternions(df, joints_dict, save=True, destfolder=None, ...)`: Computes quaternions for joint rotations.
  - `compute_joint_doubleangles(df, joints_dict, save=True, ...)`: Computes pitch/yaw double angles.
  - `plot_joint_quaternions(joint_quaternion, quats=[None], start=None, end=None)`: Plots quaternion components.
  - `compute_joint_quaternion_velocity(joint_quaternion, filter_window=3, order=1)`: Computes quaternion velocity.
  - `compute_joint_quaternion_acceleration(joint_quaternion, filter_window=3, order=2)`: Computes quaternion acceleration.
  - `_load_quaternions(destfolder, output_filename)`: Loads saved quaternions.

#### 7. `plotting.py`

- **Purpose**: Plotting utilities for kinematics.
- **Key Functions**:
  - `plot_velocity(df, df_velocity, start=None, end=None)`: Plots velocity traces alongside original positions.
  - `plot_joint_angles(joint_angle, angles=[None], start=None, end=None)`: Plots joint angles.
  - `visualize_synergies(data_reconstructed)`: Visualizes synergy reconstructions.
  - `pca_plot(...)`: Internal helper for 3D PCA reconstruction plots.
  - `plot_3d_pca_reconstruction(df, n_components, framenumber, bodyparts2plot, bp_to_connect)`: Plots 3D reconstruction from PCA.
  - `plot_umap(Y, size=5, alpha=1, color="indigo", figsize=(10,6))`: Plots UMAP embeddings (2D/3D).

#### 8. `visualization.py`

- **Purpose**: Interactive 2D/3D visualizers for motion data.
- **Key Classes**:
  - `Visualizer3D`: Full 3D visualizer with skeleton, linked 2D camera views, and slider.
  - `MinimalVisualizer3D`: Lightweight 3D visualizer.
  - `MultiVisualizer`: Synchronized multi-view visualizer.
  - `Visualizer2D`: 2D visualizer with skeleton and likelihood masking.

#### 9. `utils/auxiliaryfunctions.py`

- **Purpose**: Internal helper functions.
- **Key Functions**:
  - `read_config(configname)`: Reads DLC config YAML.
  - `check_2d_or_3d(df)`: Determines if data is 2D or 3D.
  - `jointangle_calc(pos)`: Computes joint angle from three points.
  - `jointquat_calc(pos, use4d=False)`: Computes quaternion of shortest rotation.
  - `doubleangle_calc(pos)`: Computes pitch/yaw double angles.
  - `create_empty_df(df)`: Creates empty DataFrame matching input shape.
  - Various smoothing and outlier-removal utilities.

#### 10. `version.py`

- **Purpose**: Stores package version (`__version__`, `VERSION`).

### Summary of Functional Categories

1. **Data I/O**: `load_data`, `load_c3d_data`
2. **Preprocessing**: `smooth_trajectory`
3. **Basic Kinematics**: `compute_velocity`, `compute_acceleration`, `compute_speed`
4. **Joint-Angle Analysis**: `compute_joint_angles`, `compute_joint_velocity`, `compute_joint_acceleration`
5. **Quaternion Analysis**: `compute_joint_quaternions`, `compute_joint_doubleangles`
6. **Dimensionality Reduction**: `extract_kinematic_synergies` (PCA), `compute_umap`
7. **Statistical Analysis**: `compute_correlation`, `compute_pca`
8. **Visualization**: Plotting functions and interactive visualizers.
9. **Utilities**: Config reading, angle/quaternion calculations, data reshaping.

---

## ehrapy

### Overview

ehrapy (Electronic Health Record Analysis in Python) is an extension of the Scanpy library specifically designed for the analysis of electronic health record (EHR) data. It provides EHR-specific preprocessing, analysis, and visualization tools while maintaining compatibility with Scanpy's API.

### Directory Structure

- **`__init__.py`**: Exports main API modules (`pp`, `tl`, `pl`, `io`, `data`, `anndata`, `get`).
- **`preprocessing/`**: EHR-specific preprocessing functions.
- **`tools/`**: Analysis tools including survival analysis, causal inference, and cohort tracking.
- **`plot/`**: Visualization functions for EHR data.
- **`io/`**: Input/output functions for EHR data formats.
- **`data/`**: Dataset loaders and data management.
- **`anndata/`**: Extensions to AnnData for EHR data.
- **`get/`**: Functions to extract data from AnnData objects.

### Key Modules and Functions

#### Preprocessing (`pp`)

- **Encoding**: `encode`, `encode_norm`, `encode_norm_impute` for handling categorical and numerical variables.
- **Imputation**: `impute` with various strategies (mean, median, KNN, MICE).
- **Bias Detection**: `detect_bias` for identifying potential biases in EHR data.
- **Quality Control**: `qc_metrics` for EHR-specific quality metrics.
- **Filtering**: `filter_features`, `filter_samples` for data cleaning.
- **Normalization**: `normalize` for scaling numerical features.
- **Scanpy Compatibility**: Re-exports Scanpy's preprocessing functions (`scanpy_pp`).

#### Tools (`tl`)

- **Survival Analysis**: `kaplan_meier`, `cox_ph`, `logrank_test` for time-to-event analysis.
- **Causal Inference**: `causal_inference` using DoWhy framework.
- **Cohort Tracking**: `cohort_tracker` for longitudinal patient tracking.
- **Feature Ranking**: `rank_features_groups` for identifying important features.
- **Scanpy Compatibility**: Re-exports Scanpy's tools functions (`scanpy_tl`).

#### Plotting (`pl`)

- **Survival Plots**: `kaplan_meier_plot`, `cox_ph_forestplot`.
- **Cohort Tracking**: `cohort_tracker_plot`.
- **Feature Ranking**: `rank_features_groups_dotplot`, `rank_features_groups_heatmap`, etc.
- **Missing Values**: `missing_values_barplot`, `missing_values_heatmap`, `missing_values_dendrogram`.
- **Scanpy Compatibility**: Re-exports Scanpy's plotting functions (`scanpy_pl`).

#### Input/Output (`io`)

- **Data Loading**: `read_csv`, `read_h5ad`, `read_fhir` for various EHR formats.
- **FHIR Support**: Specialized functions for Fast Healthcare Interoperability Resources (FHIR) data.

#### Data (`data`)

- **Dataset Loaders**: Functions to load example EHR datasets.
- **Data Management**: Utilities for handling EHR data structures.

#### AnnData Extensions (`anndata`)

- **Feature Specifications**: Tools for managing feature types (numerical, categorical, survival).
- **AnnData Extensions**: Enhanced AnnData functionality for EHR data.

#### Get (`get`)

- **Data Extraction**: Functions like `obs_df`, `var_df` to extract data from AnnData objects.

### Technology Stack

- **Core**: Built on top of Scanpy and AnnData.
- **Machine Learning**: Integrates with scikit-learn, lifelines (survival analysis), DoWhy (causal inference).
- **Data Formats**: Supports CSV, HDF5, FHIR, and other EHR-specific formats.
- **Visualization**: Uses Matplotlib, Seaborn, and Scanpy's plotting infrastructure.

---

## scaleSC

### Overview

scaleSC is a GPU-accelerated single-cell RNA-seq analysis toolkit designed for large-scale datasets. It provides efficient implementations of common single-cell analysis tasks, including batch correction (Harmony), marker gene detection, cluster merging, and memory-efficient data loading.

### File Structure

- **`scalesc_merged.py`**: A single merged file containing all modules of the scaleSC library. This file is generated by concatenating individual module files.

### Key Components

#### Core Classes and Functions

- **`ScaleSC`**: Main class for single-cell analysis (imported from `scalesc.pp`).
- **`AnnDataBatchReader`**: Memory-efficient batch reader for extremely large single-cell datasets, supporting chunked loading from disk or preloading on CPU/GPU.
- **`clusters_merge`**: Function for merging clusters based on marker gene expression similarity.
- **`find_markers`**: Function for identifying marker genes using NSForest-style approach with GPU-accelerated random forests.

#### GPU-Accelerated Components

- **`harmony`**: GPU implementation of the Harmony batch correction algorithm.
- **`kernels.py`**: Custom CUDA kernels for efficient sparse matrix operations (mean/variance computation, indexing, etc.).
- **`util.py`**: Utility functions including GPU memory management (`gc`), data type checking (`check_nonnegative_integers`), and sparse matrix operations.

#### Marker Gene Analysis

- **`trim_merge_marker.py`**: Implements NSForest-style marker gene detection and cluster merging.
  - `myNSForest`: GPU-accelerated random forest implementation for marker detection.
  - `specificity_score`: Computes gene specificity scores across clusters.
  - `fraction_cells`: Calculates fraction of cells expressing each gene in each cluster.
  - `find_cluster_pairs_to_merge`: Identifies clusters that should be merged based on marker gene expression.

#### Memory Management

- **`AnnDataBatchReader`**: Handles extremely large datasets by loading data in chunks, with options for preloading on CPU or GPU.
- **Memory-Efficient Operations**: Custom CUDA kernels and sparse matrix operations minimize memory usage during computation.

### Technology Stack

- **GPU Acceleration**: Uses CuPy, cuML (RAPIDS), and custom CUDA kernels for high-performance computing.
- **Single-Cell Analysis**: Built on AnnData and Scanpy, with extensions for large-scale data.
- **Batch Correction**: GPU implementation of Harmony algorithm.
- **Machine Learning**: GPU-accelerated random forests (cuML) and XGBoost for marker detection.
- **Memory Management**: Optimized for datasets that exceed GPU memory capacity through chunked processing.
