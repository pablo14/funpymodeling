# funPyModeling

Python equivalent of the R package [funModeling](https://cran.r-project.org/web/packages/funModeling/). Designed to help data scientists speed up Exploratory Data Analysis, Data Preparation, and Model Performance evaluation.

Companion package for the book [Data Science Live Book](https://livebook.datascienceheroes.com).

## Install

```bash
pip install funpymodeling
```

## Quick start

```python
import funpymodeling as fm
import seaborn as sns

iris = sns.load_dataset('iris')

# Data health check
fm.status(iris)

# Numeric profiling
fm.profiling_num(iris)

# Frequency tables and plots
fm.freq_tbl(iris, input='species')
fm.freq_plot(iris, input='species')

# Histograms for all numeric variables
fm.plot_num(iris)
```

## Functions

### Exploratory Data Analysis (`exploratory`)

| Function | Description | R equivalent |
|---|---|---|
| `status(data)` | Health check: zeros, NAs, infinites, types, unique values | `df_status` |
| `profiling_num(data)` | Numeric profiling: mean, std, percentiles, skewness, kurtosis, IQR, ranges | `profiling_num` |
| `freq_tbl(data, input)` | Frequency table (proportion 0-1) for categorical variables | `freq(..., plot=FALSE)` |
| `freq_plot(data, input)` | Horizontal bar chart of frequencies | `freq(..., plot=TRUE)` |
| `plot_num(data, bins)` | Histograms for all numerical variables | `plot_num` |
| `corr_pair(data, method)` | Pairwise correlation (R and R²) | — |
| `num_vars(data)` | Returns numeric column names | — |
| `cat_vars(data)` | Returns categorical column names | — |

### Data Preparation (`data_prep`)

| Function | Description | R equivalent |
|---|---|---|
| `equal_freq(var, n_bins)` | Equal-frequency binning | `equal_freq` |
| `tukey_outlier(x)` | Tukey outlier thresholds (IQR × 3) | `tukey_outlier` |
| `hampel_outlier(x, k_mad_value)` | Hampel outlier thresholds (median ± k×MAD) | `hampel_outlier` |
| `prep_outliers(data, input, type, method)` | Handle outliers: set_na or stop (clip) | `prep_outliers` |
| `categ_analysis(data, input, target)` | Profile categorical variable vs binary target | `categ_analysis` |
| `auto_grouping(data, input, target, n_groups)` | Reduce cardinality via KMeans clustering | `auto_grouping` |
| `discretize_get_bins(data, input, n_bins)` | Get bin thresholds for discretization | `discretize_get_bins` |
| `discretize_df(data, data_bins)` | Apply discretization bins to a DataFrame | `discretize_df` |
| `todf(data)` | Convert lists, arrays, Series to DataFrame | — |

### Target & Variable Analysis (`target_analysis`)

| Function | Description | R equivalent |
|---|---|---|
| `cross_plot(data, input, target)` | Stacked bar chart: input vs binary target | `cross_plot` |
| `var_rank_info(data, target)` | Variable ranking via Information Theory (entropy, MI, IG, gain ratio) | `var_rank_info` |

### Model Validation (`model_validation`)

| Function | Description | R equivalent |
|---|---|---|
| `gain_lift(data, score, target)` | Cumulative gain and lift chart + table | `gain_lift` |
| `coord_plot(data, group_var)` | Coordinate (parallel) plot for cluster profiling | — |

## Key differences from R version

- **Percentages as proportions**: All tables return percentages as 0-1 proportions (e.g., 0.33 instead of 33%).
- **`freq` split**: R's `freq()` is split into `freq_tbl()` (table) and `freq_plot()` (chart).
- **NA handling**: `freq_tbl` and `freq_plot` include NAs by default (`na_rm=False`), matching R's `na.rm=FALSE`.
- **Outlier thresholds**: `tukey_outlier` and `hampel_outlier` return `{'lower': ..., 'upper': ...}` dicts.

## Dependencies

- pandas, numpy, matplotlib, scikit-learn, scipy, seaborn
