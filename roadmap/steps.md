# Behapy Development Prompts - Step-by-Step

Here are detailed prompts you can give to your AI agent step by step. Each prompt is self-contained and builds upon the previous steps.

---

## Phase 1: Project Initialization

* [X] **Phase 1: Project Initialization**

### Step 1.1: Create Project Structure

* [X] Create the initial directory structure for the behapy project with the following layout:

```
behapy/
├── behapy/
│   ├── __init__.py
│   ├── _version.py
│   ├── _settings.py
│   ├── io/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   ├── tools/
│   │   └── __init__.py
│   ├── plotting/
│   │   └── __init__.py
│   ├── neighbors/
│   │   └── __init__.py
│   ├── get/
│   │   └── __init__.py
│   ├── datasets/
│   │   └── __init__.py
│   └── _core/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── docs/
├── examples/
├── data/
└── scripts/

Create empty __init__.py files for all subdirectories under behapy/.
```

### Step 1.2: Create Core Configuration Files

* [X] Create the following configuration files for the behapy project:

1. pyproject.toml with:
   * Project metadata (name: "behapy", version: "0.1.0")
   * Description: "A Scanpy-like framework for analyzing high-dimensional behavioral time series data"
   * Python version requirement: >=3.8
   * Core dependencies: anndata>=0.8.0, scanpy>=1.9.0, numpy>=1.21.0, pandas>=1.3.0, scipy>=1.7.0, scikit-learn>=1.0.0, matplotlib>=3.4.0, seaborn>=0.11.0, umap-learn>=0.5.0, numba>=0.54.0, h5py>=3.0.0, tqdm>=4.62.0
   * Optional dependencies groups: gpu, video, interactive, dev, docs
   * Build system configuration
   * Tool configurations for black, isort, pytest, mypy
2. .gitignore with standard Python patterns (pycache, *.pyc, venv/, .pytest_cache/, etc.)
3. LICENSE file (BSD-3-Clause license)
4. README.md with:
   * Project title and description
   * Installation instructions
   * Quick start example
   * Links to documentation
   * Citation information
   * License badge
5. MANIFEST.in to include LICENSE, README.md, requirements files, and data files
6. requirements.txt with core dependencies only

### Step 1.3: Create Version and Settings Files

* [X] Create the following files in the behapy/ directory:

1. behapy/_version.py:
   * Define **version** = "0.1.0"
   * Define VERSION tuple = (0, 1, 0)
2. behapy/_settings.py:
   * Create a Settings class using dataclass or similar pattern
   * Include settings for: verbosity level, plotting defaults (figsize, dpi), cache directory, number of CPUs for parallel processing, random seed
   * Create a global settings instance
   * Add functions: get_settings(), set_figure_params()
3. behapy/ **init** .py:
   * Import **version** from _version
   * Import settings from _settings
   * Create placeholder imports for submodules: io, preprocessing as pp, tools as tl, plotting as pl, datasets, neighbors, get
   * Add **all** list with main exports

---

## Phase 2: Core Data Structure

* [X] **Phase 2: Core Data Structure**

### Step 2.1: Create BehapyData Class

* [X] Create behapy/_core/_behapydata.py with a BehapyData class that wraps AnnData:

Requirements:

* Import AnnData from anndata
* Create BehapyData class that either inherits from or wraps AnnData
* Add custom **init** method that accepts:
  * X: coordinate matrix (frames × features)
  * obs: DataFrame with frame-level metadata (timestamp, video_id, frame_number)
  * var: DataFrame with feature metadata (bodypart, coordinate_axis, likelihood)
  * obsm: dict for derived representations
  * layers: dict for alternative data views (raw, smoothed, velocity)
  * uns: dict for unstructured metadata (skeleton, software_info, config)
* Add properties to access: n_frames (n_obs), n_features (n_vars)
* Add methods: copy(),  **repr** (),  **str** ()
* Add validation in **init** to check coordinate dimensions match metadata

Update behapy/_core/ **init** .py to export BehapyData.

### Step 2.2: Create Data Validation Utilities

* [X] Create behapy/utils/ **init** .py and behapy/utils/_validation.py:

In _validation.py, implement:

* validate_coords(coords): Check for valid coordinate array (numeric, 2D/3D)
* validate_likelihood(likelihood): Check likelihood values are between 0 and 1
* validate_bodyparts(bodyparts, coords): Check bodyparts match coordinate dimensions
* validate_skeleton(skeleton, bodyparts): Check skeleton references valid bodyparts
* check_bdata_is_type(bdata, BehapyData): Type checking utility

Add comprehensive docstrings with Parameters and Returns sections.

Update behapy/utils/ **init** .py to export these validation functions.

---

## Phase 3: I/O Module - Basic Readers

* [ ] **Phase 3: I/O Module - Basic Readers**

### Step 3.1: Create DeepLabCut Reader

* [ ] Create behapy/io/_dlc.py with functions to read DeepLabCut output:

Implement:

1. read_dlc_h5(filepath, animal=None):
   * Read DLC H5 file using pandas
   * Handle multi-index columns (scorer, bodyparts, coords)
   * Extract likelihood values
   * Return dictionary with: coords, likelihood, bodyparts, scorer
2. read_dlc_csv(filepath, animal=None):
   * Read DLC CSV file
   * Skip header rows appropriately
   * Parse multi-index columns
   * Return same dictionary structure as read_dlc_h5
3. read_dlc(filepath, **kwargs):
   * Auto-detect file format (.h5 or .csv)
   * Call appropriate reader
   * Convert to BehapyData object
   * Set obs with frame_number, var with bodypart/coord info
   * Store raw data in layers['raw']
   * Store metadata in uns (software='deeplabcut', scorer, filepath)
   * Return BehapyData object

Add detailed docstrings with examples.

### Step 3.2: Create SLEAP Reader

* [ ] Create behapy/io/_sleap.py with functions to read SLEAP output:

Implement:

1. read_sleap_h5(filepath):
   * Read SLEAP H5 file structure
   * Extract tracks, track_names, node_names
   * Handle multiple animals if present
   * Extract confidence scores (likelihood)
   * Return dictionary with coords, likelihood, bodyparts, track_names
2. read_sleap(filepath, animal=None, **kwargs):
   * Call read_sleap_h5
   * If animal specified, extract single animal data
   * Convert to BehapyData object with proper metadata
   * Store in layers, obsm, uns similar to read_dlc
   * Return BehapyData object

Add comprehensive docstrings and handle edge cases (missing data, multiple animals).

### Step 3.3: Create Universal Reader and Writer

* [ ] Create behapy/io/_readers.py and behapy/io/_writers.py:

In _readers.py:

1. detect_software(filepath):
   * Examine file structure/headers to auto-detect software
   * Return string: 'deeplabcut', 'sleap', 'anipose', 'c3d', or 'unknown'
2. read(filepath, software='auto', **kwargs):
   * If software='auto', call detect_software()
   * Route to appropriate reader (read_dlc, read_sleap, etc.)
   * Return BehapyData object

In _writers.py:

1. write_h5ad(bdata, filepath):
   * Convert BehapyData to AnnData if needed
   * Use anndata.write_h5ad()
2. write_csv(bdata, filepath):
   * Export coordinate matrix to CSV
   * Include bodypart and coordinate labels

Update behapy/io/ **init** .py to export: read, read_dlc, read_sleap, write_h5ad, write_csv

---

## Phase 4: Preprocessing - Quality Control

* [ ] **Phase 4: Preprocessing - Quality Control**

### Step 4.1: Create QC Metrics

* [ ] Create behapy/preprocessing/_qc.py with quality control functions:

Implement:

1. calculate_qc_metrics(bdata, likelihood_threshold=0.9):
   * Calculate per-frame metrics:
     * mean_likelihood: average likelihood across all bodyparts
     * n_low_likelihood: count of bodyparts below threshold
     * n_missing: count of missing/NaN coordinates
   * Calculate per-bodypart metrics:
     * detection_rate: fraction of frames with likelihood > threshold
     * mean_likelihood: average likelihood across frames
   * Store frame metrics in bdata.obs
   * Store bodypart metrics in bdata.var
   * Store thresholds in bdata.uns['qc']
   * Return bdata (modified in place)
2. detect_outliers(bdata, method='zscore', threshold=3.0):
   * For each bodypart, detect outlier coordinates
   * Methods: 'zscore' (Z-score), 'iqr' (interquartile range), 'speed' (velocity-based)
   * Mark outliers in new bdata.layers['outliers'] (boolean array)
   * Store detection parameters in bdata.uns
   * Return bdata

Add detailed docstrings with Parameters, Returns, and Examples sections.

### Step 4.2: Create Filtering Functions

* [ ] Create behapy/preprocessing/_filter.py with filtering functions:

Implement:

1. filter_frames(bdata, min_likelihood=0.9, min_bodyparts=None, inplace=True):
   * Filter frames based on QC metrics
   * If min_bodyparts is None, use all bodyparts
   * Remove frames where n_low_likelihood > threshold
   * Update bdata by subsetting observations
   * Return filtered bdata or None if inplace
2. filter_bodyparts(bdata, min_detection_rate=0.8, inplace=True):
   * Filter bodyparts with low detection rate
   * Remove bodyparts from var and corresponding columns from X
   * Update skeleton in uns if present
   * Return filtered bdata or None if inplace
3. interpolate_missing(bdata, method='linear', max_gap=10):
   * Interpolate missing coordinates (NaN or low likelihood)
   * Methods: 'linear', 'cubic', 'nearest'
   * Only interpolate gaps smaller than max_gap frames
   * Store interpolated data in bdata.layers['interpolated']
   * Optionally update X with interpolated data
   * Return bdata

Add comprehensive docstrings and input validation.

### Step 4.3: Create Smoothing Functions

* [ ] Create behapy/preprocessing/_smooth.py with smoothing functions:

Implement:

1. smooth_savgol(bdata, window_length=5, polyorder=2, layer=None, inplace=True):
   * Apply Savitzky-Golay filter using scipy.signal.savgol_filter
   * If layer is None, smooth bdata.X; otherwise smooth bdata.layers[layer]
   * Store smoothed data in bdata.layers['smoothed']
   * Store parameters in bdata.uns['smooth']
   * Return bdata
2. smooth_gaussian(bdata, sigma=1.0, layer=None, inplace=True):
   * Apply Gaussian smoothing using scipy.ndimage.gaussian_filter1d
   * Along time axis (axis=0)
   * Store in bdata.layers['smoothed']
   * Return bdata
3. smooth_median(bdata, window_length=5, layer=None, inplace=True):
   * Apply median filter using scipy.signal.medfilt
   * Store in bdata.layers['smoothed']
   * Return bdata
4. smooth(bdata, method='savgol', **kwargs):
   * Unified interface routing to appropriate smoothing function
   * method: 'savgol', 'gaussian', 'median'
   * Pass **kwargs to underlying function
   * Return bdata

Add detailed docstrings with usage examples for each method.

---

## Phase 5: Preprocessing - Feature Extraction (Part 1)

* [ ] **Phase 5: Preprocessing - Feature Extraction (Part 1)**

### Step 5.1: Create Distance Calculations

* [ ] Create behapy/preprocessing/_features.py with feature extraction functions:

Implement:

1. compute_distances(bdata, bodypart_pairs=None, store_as='distances'):
   * If bodypart_pairs is None, compute all pairwise distances
   * Otherwise, bodypart_pairs is list of tuples: [('nose', 'tail'), ...]
   * For each pair, compute Euclidean distance per frame
   * Create feature names: 'dist_nose_tail'
   * Store in bdata.obsm[store_as] as DataFrame
   * Store pair info in bdata.uns[f'{store_as}_pairs']
   * Return bdata
2. compute_speed(bdata, bodyparts=None, fps=30, store_as='speed'):
   * If bodyparts is None, compute for all bodyparts
   * Compute frame-to-frame displacement using np.diff
   * Multiply by fps to get speed (units/second)
   * Store in bdata.obsm[store_as] as DataFrame with columns for each bodypart
   * Store fps in bdata.uns
   * Return bdata
3. compute_acceleration(bdata, bodyparts=None, fps=30, store_as='acceleration'):
   * Compute second derivative of coordinates
   * Similar structure to compute_speed
   * Return bdata

Add comprehensive docstrings with mathematical formulas in the description.

### Step 5.2: Create Angle Calculations

* [ ] Continue in behapy/preprocessing/_features.py:

Implement:

1. compute_angles(bdata, joint_dict, store_as='angles'):
   * joint_dict format: {'joint_name': ['bodypart1', 'bodypart2', 'bodypart3']}
   * For each joint (3 bodyparts defining an angle):
     * Extract coordinates of the 3 points
     * Compute angle using vectors: angle = arccos(dot(v1, v2) / (norm(v1) * norm(v2)))
     * Store in degrees
   * Store in bdata.obsm[store_as] as DataFrame
   * Store joint_dict in bdata.uns[f'{store_as}_joints']
   * Return bdata
2. Helper function: _angle_between_points(p1, p2, p3):
   * Compute angle at p2 formed by points p1-p2-p3
   * Handle 2D and 3D coordinates
   * Return angle in degrees

Add input validation to ensure joint_dict references existing bodyparts.

### Step 5.3: Create Numba-Accelerated Feature Extraction (2D)

* [ ] Create behapy/preprocessing/_features_2d.py with Numba-accelerated functions:

Implement using @njit decorator from numba:

1. compute_distances_numba(coords, pairs_idx):
   * coords: (n_frames, n_bodyparts, 2) array
   * pairs_idx: (n_pairs, 2) array of bodypart indices
   * Return: (n_frames, n_pairs) array of distances
   * Use vectorized operations for speed
2. compute_angles_numba(coords, joints_idx):
   * coords: (n_frames, n_bodyparts, 2) array
   * joints_idx: (n_joints, 3) array of bodypart indices
   * Return: (n_frames, n_joints) array of angles
   * Compute using vector dot products
3. compute_velocity_numba(coords, fps):
   * coords: (n_frames, n_bodyparts, 2) array
   * Return: (n_frames, n_bodyparts, 2) array of velocities
   * Use np.diff and handle boundary
4. compute_speed_numba(velocity):
   * velocity: (n_frames, n_bodyparts, 2) array
   * Return: (n_frames, n_bodyparts) array of speeds
   * Compute vector magnitude

Add docstrings explaining the Numba acceleration and array shapes.

---

## Phase 6: Preprocessing - Normalization and Neighbors

* [ ] **Phase 6: Preprocessing - Normalization and Neighbors**

### Step 6.1: Create Normalization Functions

* [ ] Create behapy/preprocessing/_normalize.py:

Implement:

1. normalize_total(bdata, target_sum=1e4, layer=None, inplace=True):
   * Normalize each frame so feature values sum to target_sum
   * If layer is None, normalize bdata.X
   * Store normalized data in bdata.layers['normalized']
   * Store parameters in bdata.uns['normalize']
   * Return bdata
2. scale(bdata, zero_center=True, max_value=None, layer=None, inplace=True):
   * Z-score standardization: (X - mean) / std
   * If zero_center=False, don't subtract mean
   * If max_value is not None, clip values
   * Store scaled data in bdata.layers['scaled']
   * Store scaling parameters (mean, std) in bdata.uns['scale']
   * Return bdata
3. log_transform(bdata, base=None, layer=None, inplace=True):
   * Apply log transformation: log(X + 1) if base is None, else log_base(X + 1)
   * Store in bdata.layers['log']
   * Return bdata

Add input validation and handle edge cases (zero/negative values for log).

### Step 6.2: Create Coordinate Transformations

* [ ] Create behapy/preprocessing/_transform.py:

Implement:

1. egocentric_alignment(bdata, ref_bodypart, heading_bodypart=None, inplace=True):
   * Translate all coordinates so ref_bodypart is at origin (0,0) each frame
   * If heading_bodypart provided, also rotate so ref->heading points along x-axis
   * Apply transformation to all bodyparts
   * Store transformed data in bdata.layers['egocentric']
   * Store parameters in bdata.uns['egocentric']
   * Return bdata
2. pixel_to_real(bdata, scale_factor=None, pixel_range=None, real_range=None, inplace=True):
   * Convert pixel coordinates to real-world units
   * If scale_factor provided, multiply coordinates by it
   * If pixel_range and real_range provided, compute scale_factor
   * Update bdata.X and all layers
   * Store scale_factor in bdata.uns['scale_factor']
   * Return bdata
3. center_coordinates(bdata, center='mean', inplace=True):
   * Center coordinates around 'mean' or 'median' across all frames
   * Store original center in bdata.uns
   * Return bdata

Add detailed docstrings with coordinate system diagrams in examples.

### Step 6.3: Create Neighbor Graph Construction

* [ ] Create behapy/preprocessing/_neighbors.py:

Implement:

1. neighbors(bdata, n_neighbors=15, n_pcs=None, use_rep=None, metric='euclidean', method='umap'):
   * If n_pcs is not None, first compute PCA with n_pcs components
   * If use_rep is None, use bdata.X; else use bdata.obsm[use_rep]
   * Compute k-nearest neighbors graph
   * Methods: 'umap' (using umap.UMAP), 'sklearn' (using sklearn.neighbors)
   * Store connectivities in bdata.obsp['connectivities'] (sparse matrix)
   * Store distances in bdata.obsp['distances']
   * Store parameters in bdata.uns['neighbors']
   * Return bdata
2. Helper: _compute_connectivities_umap(X, n_neighbors):
   * Use UMAP to compute fuzzy simplicial set
   * Return connectivities sparse matrix

Import from sklearn.decomposition import PCA and sklearn.neighbors.

Update behapy/preprocessing/ **init** .py to export all preprocessing functions.

---

## Phase 7: Tools - Dimensionality Reduction

* [ ] **Phase 7: Tools - Dimensionality Reduction**

### Step 7.1: Create PCA Function

* [ ] Create behapy/tools/_embedding.py:

Implement:

1. pca(bdata, n_comps=50, zero_center=True, svd_solver='arpack', use_rep=None, layer=None):
   * If use_rep/layer specified, use that data; else use bdata.X
   * Use sklearn PCA or scanpy.pp.pca
   * Store PCA coordinates in bdata.obsm['X_pca']
   * Store loadings in bdata.varm['PCs']
   * Store variance ratio in bdata.uns['pca']['variance_ratio']
   * Store variance in bdata.uns['pca']['variance']
   * Return bdata

Add detailed docstring explaining when to use PCA for behavioral data.

### Step 7.2: Create UMAP Function

* [ ] Continue in behapy/tools/_embedding.py:

Implement:

1. umap(bdata, min_dist=0.5, spread=1.0, n_components=2, random_state=0, n_neighbors=None, use_rep='X_pca'):
   * If use_rep is 'X_pca' and not present, run pca() first
   * If n_neighbors is None, use value from bdata.uns['neighbors']
   * Use umap-learn library or scanpy.tl.umap
   * Store UMAP coordinates in bdata.obsm['X_umap']
   * Store parameters in bdata.uns['umap']
   * Return bdata
2. tsne(bdata, n_components=2, perplexity=30, early_exaggeration=12, learning_rate=1000, random_state=0, use_rep='X_pca'):
   * Similar structure to umap
   * Use sklearn.manifold.TSNE
   * Store in bdata.obsm['X_tsne']
   * Return bdata

Add notes about when to use UMAP vs t-SNE for behavioral data visualization.

---

## Phase 8: Tools - Clustering

* [ ] **Phase 8: Tools - Clustering**

### Step 8.1: Create Leiden Clustering

* [ ] Create behapy/tools/_clustering.py:

Implement:

1. leiden(bdata, resolution=1.0, random_state=0, n_iterations=-1, use_rep=None):
   * Check if neighbors graph exists in bdata.obsp['connectivities']
   * If not, raise error asking user to run pp.neighbors first
   * Use leidenalg library or scanpy.tl.leiden
   * Store cluster labels in bdata.obs['leiden']
   * Store parameters in bdata.uns['leiden']
   * Return bdata
2. louvain(bdata, resolution=1.0, random_state=0, use_rep=None):
   * Similar to leiden
   * Use louvain-community library or scanpy.tl.louvain
   * Store in bdata.obs['louvain']
   * Return bdata

Import leidenalg and python-louvain libraries (add to dependencies).

### Step 8.2: Create HDBSCAN Clustering

* [ ] Continue in behapy/tools/_clustering.py:

Implement:

1. hdbscan(bdata, min_cluster_size=10, min_samples=None, metric='euclidean', cluster_selection_epsilon=0.0, use_rep='X_pca'):
   * If use_rep not in bdata.obsm, raise error
   * Use hdbscan library
   * Fit HDBSCAN on bdata.obsm[use_rep]
   * Store labels in bdata.obs['hdbscan']
   * Store cluster probabilities in bdata.obs['hdbscan_probabilities']
   * Store parameters and clusterer object in bdata.uns['hdbscan']
   * Return bdata
2. hierarchical_clustering(bdata, n_clusters=None, distance_threshold=None, linkage='ward', use_rep='X_pca'):
   * Use sklearn AgglomerativeClustering
   * Either n_clusters or distance_threshold must be specified
   * Store labels in bdata.obs['hierarchical']
   * Return bdata

Add hdbscan to requirements.txt.

Update behapy/tools/ **init** .py to export clustering functions.

---

## Phase 9: Plotting - Basic Visualizations

* [ ] **Phase 9: Plotting - Basic Visualizations**

### Step 9.1: Create Trajectory Plots

* [ ] Create behapy/plotting/_trajectory.py:

Implement:

1. trajectory(bdata, bodypart, color_by='time', start=None, end=None, alpha=0.8, ax=None, show=True, **kwargs):
   * Extract coordinates for specified bodypart from bdata
   * If start/end specified, subset frames
   * Create 2D plot (x, y) or 3D plot if coordinates are 3D
   * Color by: 'time' (frame number), 'speed', or any column in bdata.obs
   * Use matplotlib scatter or plot with colormap
   * Add colorbar if color_by is continuous
   * If ax is None, create new figure
   * If show=True, call plt.show()
   * Return ax
2. heatmap(bdata, bodypart, bins=50, cmap='viridis', ax=None, show=True):
   * Create 2D histogram of bodypart positions
   * Use np.histogram2d or plt.hexbin
   * Add colorbar showing density
   * Return ax

Add comprehensive docstrings with example usage.

### Step 9.2: Create Embedding Plots

* [ ] Create behapy/plotting/_embedding.py:

Implement:

1. embedding(bdata, basis='umap', color=None, size=5, alpha=0.8, legend_loc='right', ax=None, show=True, **kwargs):
   * basis: 'umap', 'tsne', 'pca'
   * Check if f'X_{basis}' exists in bdata.obsm
   * Extract 2D or 3D coordinates
   * If color is None, use single color
   * If color is string, look up in bdata.obs or bdata.var
   * Create scatter plot with appropriate coloring
   * If color is categorical, add legend
   * If color is continuous, add colorbar
   * Return ax
2. umap(bdata, color=None, **kwargs):
   * Wrapper calling embedding(bdata, basis='umap', color=color, **kwargs)
3. tsne(bdata, color=None, **kwargs):
   * Wrapper calling embedding(bdata, basis='tsne', color=color, **kwargs)
4. pca(bdata, components=[0, 1], color=None, **kwargs):
   * Plot specified PCA components
   * components: list of 2 or 3 component indices
   * Return ax

Add support for multiple colors (list of columns) to create multi-panel plots.

### Step 9.3: Create PCA Variance Plot

* [ ] Continue in behapy/plotting/_embedding.py:

Implement:

1. pca_variance_ratio(bdata, n_pcs=50, log=False, ax=None, show=True):
   * Check if PCA has been computed
   * Extract variance_ratio from bdata.uns['pca']
   * Create scree plot (bar or line plot)
   * If log=True, use log scale on y-axis
   * Add cumulative variance line
   * Add horizontal line at 0.8 cumulative variance
   * Return ax

Add docstring explaining how to interpret variance plots for behavioral data.

Update behapy/plotting/ **init** .py to export trajectory and embedding functions.

---

## Phase 10: Plotting - Behavior Visualizations

* [ ] **Phase 10: Plotting - Behavior Visualizations**

### Step 10.1: Create Ethogram Plot

* [ ] Create behapy/plotting/_behavior.py:

Implement:

1. ethogram(bdata, behavior_key='behavior', start=None, end=None, fps=30, ax=None, show=True, **kwargs):
   * Extract behavior labels from bdata.obs[behavior_key]
   * Subset frames if start/end specified
   * Create horizontal bar plot showing behavior over time
   * X-axis: time (in seconds, using fps)
   * Y-axis: single row colored by behavior
   * Use distinct colors for each behavior
   * Add legend mapping colors to behavior names
   * Return ax
2. behavior_pie(bdata, behavior_key='behavior', ax=None, show=True):
   * Count occurrences of each behavior
   * Create pie chart
   * Add percentages to labels
   * Return ax
3. bout_distribution(bdata, behavior_key='behavior', bins=30, log=False, ax=None, show=True):
   * Extract behavior labels
   * Identify continuous bouts (consecutive frames with same label)
   * Compute bout durations
   * Create histogram of bout durations per behavior
   * If log=True, use log scale
   * Return ax

Add helper function to identify bouts: _extract_bouts(labels) returning list of (behavior, start, duration).

### Step 10.2: Create Feature Ranking Plots

* [ ] Create behapy/plotting/_features.py:

Implement:

1. rank_features_groups(bdata, groupby='behavior', n_features=20, method='wilcoxon', ax=None, show=True):
   * Check if rank_features_groups has been run (look for bdata.uns['rank_features_groups'])
   * Extract top n_features for each group
   * Create horizontal bar plot or dot plot showing feature scores
   * Return ax

Note: This requires implementing tl.rank_features_groups first (defer until Phase 11).

2. heatmap(bdata, features=None, groupby='behavior', standard_scale='var', cmap='viridis', figsize=(8, 6), show=True):
   * If features is None, use all features
   * Group frames by groupby column
   * Compute mean feature values per group
   * Create clustered heatmap using seaborn
   * If standard_scale='var', scale features (Z-score)
   * Return fig, ax

Update behapy/plotting/ **init** .py to export behavior and feature plotting functions.

---

## Phase 11: Tools - Marker Feature Detection

* [ ] **Phase 11: Tools - Marker Feature Detection**

### Step 11.1: Create Differential Feature Analysis

* [ ] Create behapy/tools/_markers.py:

Implement:

1. rank_features_groups(bdata, groupby, groups='all', reference='rest', method='wilcoxon', n_features=100, use_rep=None):

   * groupby: column in bdata.obs with group labels (e.g., 'behavior')
   * groups: 'all' or list of specific groups to test
   * reference: 'rest' (compare each group to all others) or specific group name
   * method: 'wilcoxon' (Wilcoxon rank-sum), 't-test', or 'logreg'

   For each group:

   * Extract features for frames in that group
   * Extract features for reference frames
   * Perform statistical test
   * Rank features by score (test statistic)
   * Store top n_features

   Store results in bdata.uns['rank_features_groups']:

   * 'params': parameters used
   * 'names': top feature names per group (structured array)
   * 'scores': test statistics per feature per group
   * 'pvals': p-values per feature per group (if applicable)

   Return bdata
2. Helper functions:

   * _wilcoxon_test(X_group, X_reference): Wilcoxon rank-sum test
   * _ttest(X_group, X_reference): t-test
   * _logreg(X, y): Logistic regression coefficients

Use scipy.stats for statistical tests.

Update behapy/tools/ **init** .py to export rank
