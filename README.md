# funPyModeling

Python equivalent of the R package [funModeling](https://cran.r-project.org/web/packages/funModeling/vignettes/funModeling_quickstart.html). Designed for data scientists and teachers to speed up **Exploratory Data Analysis**, **Data Preparation**, and **Model Performance** evaluation.

Companion package for the book [Data Science Live Book](https://livebook.datascienceheroes.com).

## Install

```bash
pip install funpymodeling
```

## Opening the black-box

Some functions have comments in the source code so users can open the black-box and learn how they were developed, or fine-tune/improve any of them.

All functions are well documented. Access the documentation via `help(function_name)`.

---

## Quick-start

```python
from funpymodeling import *
import seaborn as sns

# Load sample data
iris = sns.load_dataset('iris')
```

---

## Exploratory Data Analysis

### `status`: Dataset health status

For each variable: quantity and percentage of zeros, NAs, and infinite values. Also reports data type and number of unique values.

```python
status(iris)
```

### `plot_num`: Plotting distributions for numerical variables

Histograms for all numerical variables in a single figure.

```python
plot_num(iris)
plot_num(iris, bins=20)
```

### `profiling_num`: Calculating several statistics for numerical variables

Returns mean, std_dev, variation_coef, percentiles (p_01 to p_99), skewness, kurtosis, IQR, range_98 and range_80.

```python
profiling_num(iris)
```

### `freq_tbl` / `freq_plot`: Frequency distributions for categoric variables

R's `freq()` is split into two functions: `freq_tbl()` for the table and `freq_plot()` for the chart.

```python
# Table
freq_tbl(iris, input='species')

# Plot
freq_plot(iris, input='species')
```

---

## Correlation

### `var_rank_info`: Correlation based on information theory

Variable importance ranking based on Information Theory. Computes entropy (en), mutual information (mi), information gain (ig), and gain ratio (gr).

```python
from funpymodeling import var_rank_info

var_rank_info(heart_disease, target='has_heart_disease')
```

### `cross_plot`: Distribution plot between input and target variable

Shows how the input variable relates to the target, getting the likelihood rates for each bin/bucket. Supports `plot_type='both'` (default), `'percentual'`, or `'quantity'`.

```python
# Single variable
cross_plot(heart_disease, input='age', target='has_heart_disease')

# Multiple variables at once
cross_plot(heart_disease, input=['age', 'chest_pain'], target='has_heart_disease')
```

### `plotar`: Boxplot and density histogram between input and target variables

Useful to explain and report whether a variable is important or not.

```python
# Boxplot
plotar(heart_disease, input=['age', 'oldpeak'],
       target='has_heart_disease', plot_type='boxplot')

# Density histogram
plotar(heart_disease, input='age',
       target='has_heart_disease', plot_type='histdens')
```

Notes:
- `input` must be numeric and `target` must be categorical.
- `target` can be multi-class (not only binary).

### `categ_analysis`: Quantitative analysis for binary outcome

Numerical analysis of a binary target based on a categorical input variable: representativeness (`perc_rows`) and accuracy (`perc_target`) of each category.

```python
categ_analysis(data_country, input='country', target='has_flu')
```

---

## Data Preparation

### `equal_freq`: Convert numeric variable to categoric

Equal-frequency binning. Splits the variable into `n_bins` segments of approximately equal size.

```python
equal_freq(iris['sepal_length'], n_bins=5)
```

### `discretize_get_bins` / `discretize_df`: Data discretization

Two-step discretization: first get the bins, then apply them. Useful for applying the same bins to train and test sets.

```python
d_bins = discretize_get_bins(heart_disease, input=['age', 'oldpeak'], n_bins=5)
heart_disease_disc = discretize_df(heart_disease, d_bins)
```

### `convert_df_to_categoric`: Convert every column to categorical

Numeric variables are binned using equal-frequency; character/factor variables are cast to string.

```python
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True).frame

iris_cat = convert_df_to_categoric(iris, n_bins=5)
```

### `range01`: Scales variable into the 0 to 1 range

Min-max normalization. Converts a numeric vector to a scale from 0 to 1.

```python
range01(heart_disease['oldpeak'])
```

### `auto_grouping`: Automatic variable grouping

Reduces cardinality of a categorical variable using KMeans clustering on the profiling metrics (`perc_rows`, `perc_target`).

```python
res = auto_grouping(data_country, input='country', target='has_flu', n_groups=3)
res['df_equivalence']   # mapping from original to new categories
res['recateg_results']  # profiling of the new groups
```

---

## Outliers Data Preparation

### `tukey_outlier` / `hampel_outlier`: Gets outlier thresholds

```python
tukey_outlier(heart_disease['age'])
# {'lower': ..., 'upper': ...}

hampel_outlier(heart_disease['age'])
# {'lower': ..., 'upper': ...}
```

### `prep_outliers`: Prepare outliers in a data frame

Two modes: `type='stop'` clips values at the threshold (recommended for predictive modeling); `type='set_na'` converts outliers to NaN (recommended for statistical analysis).

Three methods: `'tukey'`, `'hampel'`, `'bottom_top'`.

```python
# Clip outliers using Tukey method
prep_outliers(heart_disease, input=['age', 'oldpeak'], type='stop', method='tukey')

# Set outliers to NA using Hampel method
prep_outliers(heart_disease, input=['age'], type='set_na', method='hampel')

# Bottom/top percentile
prep_outliers(heart_disease, input=['age'], type='stop',
              method='bottom_top', top_percent=0.01, bottom_percent=0.01)
```

---

## Predictive Model Performance

### `gain_lift`: Gain and lift performance curve

Higher values at the beginning of the population implies a better model.

```python
gain_lift(data=scored_data, score='score', target='has_heart_disease')
```

### `coord_plot`: Coordinate plot (clustering models)

Coordinate (parallel) plot for cluster profiling. Returns original and min-max normalized tables.

```python
coord_plot(iris, group_var='species')
```

---

## All Functions Reference

### Exploratory Data Analysis

| Function | Description | R equivalent |
|---|---|---|
| `status(data)` | Health check: zeros, NAs, infinites, types, unique values | `df_status` |
| `profiling_num(data)` | Numeric profiling: mean, std, percentiles, skewness, kurtosis | `profiling_num` |
| `freq_tbl(data, input)` | Frequency table for categorical variables | `freq` |
| `freq_plot(data, input)` | Horizontal bar chart of frequencies | `freq` |
| `plot_num(data, bins)` | Histograms for all numerical variables | `plot_num` |
| `corr_pair(data, method)` | Pairwise correlation (R and R²) | `correlation_table` |
| `num_vars(data)` | Returns numeric column names | — |
| `cat_vars(data)` | Returns categorical column names | — |

### Data Preparation

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
| `range01(x)` | Min-max normalization to [0, 1] | `range01` |
| `convert_df_to_categoric(data, n_bins)` | Convert all columns to categorical | `convert_df_to_categoric` |
| `todf(data)` | Convert lists, arrays, Series to DataFrame | — |

### Target & Variable Analysis

| Function | Description | R equivalent |
|---|---|---|
| `cross_plot(data, input, target)` | Stacked bar chart: input vs binary target | `cross_plot` |
| `var_rank_info(data, target)` | Variable ranking via Information Theory | `var_rank_info` |
| `plotar(data, input, target, plot_type)` | Boxplot or density histogram grouped by target | `plotar` |

### Model Validation

| Function | Description | R equivalent |
|---|---|---|
| `gain_lift(data, score, target)` | Cumulative gain and lift chart + table | `gain_lift` |
| `coord_plot(data, group_var)` | Coordinate (parallel) plot for cluster profiling | `coord_plot` |

## Key differences from R version

- **`freq` split**: R's `freq()` is split into `freq_tbl()` (table) and `freq_plot()` (chart).
- **Percentages as proportions**: Frequency tables return percentages as 0-1 proportions (e.g., 0.33 instead of 33%).
- **NA handling**: `freq_tbl` and `freq_plot` include NAs by default (`na_rm=False`).
- **Outlier thresholds**: `tukey_outlier` and `hampel_outlier` return `{'lower': ..., 'upper': ...}` dicts.
- **Parameter naming**: Uses `input=` instead of R's variable-specific parameter names.

## Dependencies

pandas, numpy, matplotlib, scikit-learn, scipy, seaborn
