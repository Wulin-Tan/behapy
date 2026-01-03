# Changelog

## [0.4.0] - 2026-01-03

### Added
- **Statistical Testing Suite** - Complete statistical comparison tools
  - `bp.tl.compare_groups()` - Multi-group comparisons with permutation/parametric tests and FDR correction
  - `bp.tl.test_transition_matrix()` - Element-wise transition matrix comparison using permutation tests
  - `bp.tl.test_behavior_frequency()` - Chi-square and Fisher's exact tests
  - `bp.tl.test_bout_metrics()` - Compare bout duration/count/latency
  - `bp.tl.compute_effect_size()` - Cohen's d and Hedges' g with bootstrap CI
  - `bp.tl.bootstrap_ci()` - Generic bootstrap confidence interval utility
  - `bp.pl.effect_sizes()` - Forest plots, volcano plots, bar charts
  - `bp.pl.statistical_summary()` - Multi-panel figure showing multiple comparisons
  - Enhanced `bp.pl.transition_matrix()` with significance marker support

### Changed
- Added `statsmodels` dependency for multiple comparison correction

### Documentation
- New tutorial: `examples/statistical_testing.ipynb`
- Comprehensive docstrings for all statistical functions
