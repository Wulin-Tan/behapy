# Behapy Project File Tree

```text
.
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── MANIFEST.in
├── README.md
├── behapy
│   ├── __init__.py
│   ├── _core
│   │   ├── __init__.py
│   │   └── _behapydata.py
│   ├── _settings.py
│   ├── _version.py
│   ├── datasets
│   │   ├── __init__.py
│   │   └── _synthetic.py
│   ├── get
│   │   └── __init__.py
│   ├── io
│   │   ├── __init__.py
│   │   ├── _dlc.py
│   │   ├── _readers.py
│   │   ├── _sleap.py
│   │   └── _writers.py
│   ├── neighbors
│   │   └── __init__.py
│   ├── plotting
│   │   ├── __init__.py
│   │   ├── _behavior.py
│   │   ├── _embedding.py
│   │   ├── _features.py
│   │   ├── _statistics.py
│   │   ├── _temporal.py
│   │   └── _trajectory.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── _features.py
│   │   ├── _features_2d.py
│   │   ├── _filter.py
│   │   ├── _neighbors.py
│   │   ├── _neighbors_annoy.py
│   │   ├── _normalize.py
│   │   ├── _qc.py
│   │   ├── _smooth.py
│   │   └── _transform.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── _cluster_utils.py
│   │   ├── _clustering.py
│   │   ├── _embedding.py
│   │   ├── _markers.py
│   │   ├── _statistics.py
│   │   └── _temporal.py
│   └── utils
│       ├── __init__.py
│       └── _validation.py
├── examples
│   └── statistical_testing.ipynb
├── pyproject.toml
├── requirements.txt
├── roadmap
│   ├── behapy_summary.md
│   ├── behapy_tree.md
│   ├── reference_summary.md
│   └── steps.md
├── scripts
│   ├── pyrat_info.py
│   ├── setup_pyrat_test_data.py
│   ├── test_all_pyrat_files.py
│   ├── test_cluster_merging.py
│   ├── test_features.py
│   ├── test_neighbors_speed.py
│   ├── test_plot_performance.py
│   ├── test_pyrat_load.py
│   ├── test_pyrat_pipeline.py
│   └── test_temporal_analysis.py
└── tests
    ├── __init__.py
    ├── test_core.py
    ├── test_extended.py
    ├── test_phase6_11.py
    ├── test_preprocessing.py
    ├── test_statistics.py
    └── test_tools.py
```

*Note: Some deeply nested or redundant files (like `__pycache__`, `.git`, `.mypy_cache`) are excluded for clarity. Data and log directories are omitted.*
