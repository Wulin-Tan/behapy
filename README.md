# behapy

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Scanpy-like framework for analyzing high-dimensional behavioral time series data.

## Features

- **Data Loading**: Import from DeepLabCut, SLEAP, and custom formats
- **Behavior Analysis**:
  - Bout detection and statistics
  - Transition matrix computation
  - **Statistical testing** (permutation, t-test, Mann-Whitney, Chi-square) NEW
  - **Effect size calculations** (Cohen's d, Hedges' g) NEW
- **Visualization**:
  - Ethograms, transition matrices, bout distributions
  - **Statistical summary plots and significance overlays** NEW

## Key Features

- **Scanpy-like API**: Familiar modular structure (`bh.io`, `bh.pp`, `bh.tl`, `bh.pl`).
- **AnnData-based**: Leverages the powerful `AnnData` structure for behavioral time series.
- **Pose Estimation Integration**: Native support for DeepLabCut and SLEAP outputs.
- **Numba-Accelerated**: Fast feature extraction for 2D pose data.
- **Comprehensive Workflow**: From QC and filtering to clustering and visualization.

## Modules

### Input/Output (`bh.io`)
- `read_dlc()`: Read DeepLabCut H5 or CSV files.
- `read_sleap()`: Read SLEAP H5 files.
- `read()`: Universal reader with auto-detection.
- `write_h5ad()`: Save BehapyData objects.

### Preprocessing (`bh.pp`)
- **QC & Filtering**: `calculate_qc_metrics()`, `filter_frames()`, `filter_bodyparts()`, `interpolate_missing()`.
- **Smoothing**: `smooth()`, `smooth_savgol()`, `smooth_gaussian()`, `smooth_median()`.
- **Feature Extraction**: `compute_distances()`, `compute_speed()`, `compute_acceleration()`, `compute_jerk()`, `compute_angles()`, `compute_angular_velocity()`.
- **Transformations**: `egocentric_alignment()`, `pixel_to_real()`, `center_coordinates()`.
- **Normalization**: `normalize_total()`, `scale()`, `log_transform()`, `quantile_normalization()`.
- **Neighbors**: `neighbors()` graph construction.

### Tools (`bh.tl`)
- **Embeddings**: `pca()`, `umap()`, `tsne()`.
- **Clustering**: `leiden()`, `louvain()`, `hdbscan()`, `kmeans()`, `hierarchical_clustering()`.
- **Markers**: `rank_features_groups()` for differential feature analysis.

### Plotting (`bh.pl`)
- **Embeddings**: `umap()`, `tsne()`, `pca()`, `pca_variance_ratio()`.
- **Trajectories**: `trajectory()`, `trajectory_heatmap()`.
- **Behavior**: `ethogram()`, `behavior_pie()`, `bout_distribution()`.
- **Features**: `feature_group_heatmap()`, `feature_time_heatmap()`, `rank_features_groups()`.

## Installation

```bash
pip install behapy
```

### Installation Guide for Users in China

If you are in China, it is recommended to use the PyPI mirror provided by Tsinghua University to accelerate downloads:

```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple behapy
```

If you need to manually install dependencies, you can use the following command:

```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    anndata scanpy numpy pandas scipy scikit-learn \
    matplotlib seaborn umap-learn==0.5.5 numba h5py \
    tqdm leidenalg==0.10.0 python-igraph python-louvain hdbscan
```

## Quick Start

```python
import behapy as bh

# Read DeepLabCut data
bdata = bh.io.read_dlc("path/to/dlc_output.h5")

# Preprocessing
bh.pp.calculate_qc_metrics(bdata)
bh.pp.filter_frames(bdata, min_likelihood=0.9)
bh.pp.smooth(bdata, method='savgol')

# Feature Extraction
bh.pp.compute_distances(bdata)
bh.pp.compute_speed(bdata)

# Analysis
bh.pp.neighbors(bdata)
bh.tl.umap(bdata)
bh.tl.leiden(bdata)

# Visualization
bh.pl.umap(bdata, color='leiden')
```

## Usage Example: Statistical Comparison

```python
import behapy as bp

# Load and compute transitions
control = bp.read_deeplabcut('control.h5')
treatment = bp.read_deeplabcut('treatment.h5')
bp.tl.compute_transitions(control)
bp.tl.compute_transitions(treatment)

# Statistical comparison
results = bp.tl.test_transition_matrix(control, treatment)
print(results[results['significant']])

# Visualize with significance
bp.pl.transition_matrix(control, comparison_bdata=treatment, show_significance=True)
```

## Documentation

Coming soon.

## Acknowledgment

This project was developed based on specifications provided in the [steps.md](file:///Users/wulintan/Nutstore%20Files/Nutstore_Onedrive_SYSU/research/code_pool/behapy_project/02_behapy_project_trae/plans/steps.md) and [reference_summary.md](file:///Users/wulintan/Nutstore%20Files/Nutstore_Onedrive_SYSU/research/code_pool/behapy_project/02_behapy_project_trae/plans/reference_summary.md) guidelines. It references and builds upon the following materials and open-source projects:

- [A-SOID](https://github.com/YttriLab/A-SOID) (referenced for multi-animal pose data handling)
- [DLC2Kinematics](https://github.com/AdaptiveMotorControlLab/DLC2Kinematics) (referenced for kinematic feature extraction)
- [PyRAT](https://github.com/D-Seabra/PyRAT) (referenced for behavioral analysis patterns)
- [ehrapy](https://github.com/theislab/ehrapy) (referenced for AnnData-based clinical/time-series data structures)
- [Scanpy](https://github.com/scverse/scanpy) (primary architectural inspiration)
- [AnnData](https://github.com/scverse/anndata) (core data structure)

## Citation

If you use behapy in your research, please cite:
Tan, W. (2026). behapy: A Scanpy-like framework for behavioral analysis.

## Author

**Wulin Tan**  
Sun Yat-sen University  
Email: wulin.tan9527@gmail.com  
GitHub: `https://github.com/Wulin-Tan`

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.
