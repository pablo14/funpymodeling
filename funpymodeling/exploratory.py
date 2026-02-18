import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .data_prep import todf


def status(data):
    """
    For each variable it returns: Quantity and percentage of zeros (q_zeros and p_zeros respectively).
    Same metrics for NA values (q_nan/p_nan), and infinite values (q_inf/p_inf).
    Last two columns indicates data type and quantity of unique values.
    Equivalent to funModeling::df_status in R.

    Parameters:
    -----------
    data: It can be a dataframe or a single column, 1D or 2D numpy array. It uses the todf() function.

    Returns:
    --------
    A pandas dataframe containing the status metrics for each input variable.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> status(iris)
    """
    data2 = todf(data)
    tot_rows = len(data2)

    d2 = data2.isnull().sum().reset_index()
    d2.columns = ['variable', 'q_nan']
    d2['p_nan'] = round(d2['q_nan'] / tot_rows, 3)
    d2['q_zeros'] = (data2 == 0).sum().values
    d2['p_zeros'] = round(d2['q_zeros'] / tot_rows, 3)
    d2['q_inf'] = data2.apply(lambda x: np.isinf(x).sum() if np.issubdtype(x.dtype, np.number) else 0).values
    d2['p_inf'] = round(d2['q_inf'] / tot_rows, 3)
    d2['type'] = [str(x) for x in data2.dtypes.values]
    d2['unique'] = data2.nunique().values

    return d2


def corr_pair(data, method='pearson'):
    """
    Calculate the correlations among all numeric features.
    Non-numeric are excluded since it uses the `corr` pandas function.

    Parameters:
    -----------
    data: pandas data containing the variables to calculate the correlation
    method: 'pearson' as default, same as `corr` function in pandas.

    Returns:
    --------
    A pandas dataframe containing pairwise correlation, R and R2 statistics.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> corr_pair(iris)
    """
    data2 = todf(data)
    d_cor = data2.corr(method)
    d_cor2 = d_cor.reset_index()
    d_long = d_cor2.melt(id_vars='index')
    d_long.columns = ['v1', 'v2', 'R']
    d_long['R'] = round(d_long['R'], 4)
    d_long['R2'] = round(d_long['R'] ** 2, 4)
    d_long2 = d_long.query("v1 != v2")
    return d_long2


def num_vars(data, exclude_var=None):
    """
    Returns the numeric variable names.

    Parameters:
    -----------
    data: pandas dataframe
    exclude_var: list of variable names to exclude from the result

    Returns:
    --------
    A pandas Index with all the numeric variable names.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> num_vars(iris)
    """
    num_v = data.select_dtypes(include=['int64', 'float64']).columns
    if exclude_var is not None:
        num_v = num_v.drop(exclude_var)
    return num_v


def cat_vars(data, exclude_var=None):
    """
    Returns the categoric variable names.

    Parameters:
    -----------
    data: pandas dataframe
    exclude_var: list of variable names to exclude from the result

    Returns:
    --------
    A pandas Index with all the categoric variable names.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> cat_vars(iris)
    """
    cat_v = data.select_dtypes(include=['object', 'category', 'string']).columns
    if exclude_var is not None:
        cat_v = cat_v.drop(exclude_var)
    return cat_v


def profiling_num(data):
    """
    Get a metric table with many indicators for all numerical variables,
    automatically skipping the non-numerical variables.
    Metrics: mean, std_dev, variation_coef, percentiles (p_01 to p_99),
    skewness, kurtosis, iqr, range_98, range_80.
    All NA values will be skipped from calculations.
    Equivalent to funModeling::profiling_num in R.

    Parameters:
    -----------
    data: pandas series/dataframe, numpy 1D/2D array

    Returns:
    --------
    A dataframe in which each row is an input variable, and each column a statistic.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> profiling_num(iris)
    """
    data = todf(data)
    d = data[num_vars(data)]

    if d.shape[1] == 0:
        raise ValueError("None of the input variables are numeric.")

    des1 = pd.DataFrame({
        'mean': round(d.mean(), 4),
        'std_dev': round(d.std(), 4)
    })
    des1['variation_coef'] = round(des1['std_dev'] / des1['mean'], 4)

    d_quant = d.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose()
    d_quant.columns = ['p_01', 'p_05', 'p_25', 'p_50', 'p_75', 'p_95', 'p_99']
    d_quant = d_quant.round(4)

    des2 = des1.join(d_quant, how='outer')
    des2['skewness'] = round(d.skew(), 4)
    des2['kurtosis'] = round(d.kurtosis(), 4)
    des2['iqr'] = round(d.quantile(0.75) - d.quantile(0.25), 4)

    q01 = d.quantile(0.01)
    q99 = d.quantile(0.99)
    q10 = d.quantile(0.10)
    q90 = d.quantile(0.90)
    des2['range_98'] = [f'[{round(q01[c], 2)}, {round(q99[c], 2)}]' for c in d.columns]
    des2['range_80'] = [f'[{round(q10[c], 2)}, {round(q90[c], 2)}]' for c in d.columns]

    des2['variable'] = des2.index
    des2 = des2.reset_index(drop=True)

    col_order = ['variable', 'mean', 'std_dev', 'variation_coef',
                 'p_01', 'p_05', 'p_25', 'p_50', 'p_75', 'p_95', 'p_99',
                 'skewness', 'kurtosis', 'iqr', 'range_98', 'range_80']
    return des2[col_order]


# ---------------------------------------------------------------
# freq_tbl internals
# ---------------------------------------------------------------
def _freq_tbl_logic(var, name):
    """Internal function for freq_tbl."""
    freq = var.value_counts(dropna=False).reset_index()
    freq.columns = [name, 'frequency']
    # Replace NaN index values with 'NA' string for display
    # Convert to object first to handle categorical dtypes where 'NA' may not be in categories
    if hasattr(freq[name], 'cat'):
        freq[name] = freq[name].astype(object)
    freq[name] = freq[name].fillna('NA')
    freq = freq.sort_values('frequency', ascending=False).reset_index(drop=True)
    total = freq['frequency'].sum()
    freq['percentage'] = round(freq['frequency'] / total, 4)
    freq['cumulative_perc'] = round(freq['percentage'].cumsum(), 4)
    freq.loc[freq.index[-1], 'cumulative_perc'] = 1.0
    return freq


def freq_tbl(data, input=None, na_rm=False):
    """
    Frequency table for categorical variables. Retrieves the frequency,
    percentage (as proportion 0 to 1) and cumulative percentage.
    Equivalent to the table output of funModeling::freq in R.

    Parameters:
    -----------
    data: pandas series/dataframe, numpy 1D/2D array
    input: string or list of strings with column names.
           If None, runs for all categorical variables.
    na_rm: if True, excludes NA values from the analysis. False by default.

    Returns:
    --------
    If a single variable is passed, returns the frequency table DataFrame.
    If multiple variables, prints each table and returns None.

    Example:
    --------
    >> import seaborn as sns
    >> tips = sns.load_dataset('tips')
    >> freq_tbl(tips)
    >> freq_tbl(tips, input='sex')
    >> freq_tbl(tips, input=['sex', 'smoker'])
    """
    data = todf(data)

    if input is not None:
        if isinstance(input, str):
            cols = [input]
        else:
            cols = list(input)
    else:
        cols = list(cat_vars(data))
        if len(cols) == 0:
            return 'No categorical variables to analyze.'

    if len(cols) == 1:
        var = data[cols[0]].dropna() if na_rm else data[cols[0]]
        return _freq_tbl_logic(var, name=cols[0])
    else:
        for col in cols:
            var = data[col].dropna() if na_rm else data[col]
            print(_freq_tbl_logic(var, name=col))
            print('\n----------------------------------------------------------------\n')
        return f"Variables processed: {', '.join(cols)}"


# ---------------------------------------------------------------
# freq_plot
# ---------------------------------------------------------------
def freq_plot(data, input=None, na_rm=False):
    """
    Plot horizontal bar charts of frequencies for categorical variables.
    Equivalent to the plotting component of funModeling::freq in R.

    Parameters:
    -----------
    data: pandas series/dataframe, numpy 1D/2D array
    input: string or list of strings with column names.
           If None, runs for all categorical variables.
    na_rm: if True, excludes NA values. False by default.

    Returns:
    --------
    None (displays plots).

    Example:
    --------
    >> import seaborn as sns
    >> tips = sns.load_dataset('tips')
    >> freq_plot(tips)
    >> freq_plot(tips, input='sex')
    """
    data = todf(data)

    if input is not None:
        if isinstance(input, str):
            cols = [input]
        else:
            cols = list(input)
    else:
        cols = list(cat_vars(data))
        if len(cols) == 0:
            print('No categorical variables to plot.')
            return

    for col in cols:
        var = data[col].dropna() if na_rm else data[col]
        _freq_plot_logic(var, col)


def _freq_plot_logic(var, name):
    """Internal function to generate a single freq plot."""
    freq = var.value_counts(dropna=False)
    total = len(var)
    percentages = round(freq / total, 4)

    fig, ax = plt.subplots(figsize=(7, max(2, 0.4 * len(freq))))

    colors = ['#5DA5DA', '#60BD68', '#F17CB0', '#B2912F',
              '#B276B2', '#DECF3F', '#F15854', '#4D4D4D']
    bar_colors = [colors[i % len(colors)] for i in range(len(freq))]

    bars = ax.barh(range(len(freq)), freq.values,
                   color=bar_colors, edgecolor='black', linewidth=0.5)

    max_value = freq.values.max()
    ax.set_xlim(0, max_value * 1.35)

    for i, (count, pct) in enumerate(zip(freq.values, percentages.values)):
        label = f'{count} ({round(pct * 100, 1)}%)'
        if count > max_value * 0.6:
            ax.text(count * 0.98, i, label, va='center', ha='right',
                    fontsize=10, color='white', weight='bold')
        else:
            ax.text(count, i, f' {label}', va='center', ha='left',
                    fontsize=10)

    labels = [str(x) if not pd.isna(x) else 'NA' for x in freq.index]
    ax.set_yticks(range(len(freq)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Frequency / (Percentage %)', fontsize=12)
    ax.set_ylabel(name, fontsize=12)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# plot_num
# ---------------------------------------------------------------
def plot_num(data, bins=10):
    """
    Plot histograms for all numerical variables in a single figure.
    NA values are automatically excluded.
    Equivalent to funModeling::plot_num in R.

    Parameters:
    -----------
    data: pandas dataframe
    bins: number of bins for each histogram, 10 by default

    Returns:
    --------
    None (displays the plot).

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> plot_num(iris)
    >> plot_num(iris, bins=20)
    """
    data = todf(data)
    num_v = num_vars(data)

    if len(num_v) == 0:
        print('No numerical variables to plot.')
        return

    n = len(num_v)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))

    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n > 1 else axes

    for i, col in enumerate(num_v):
        ax = axes[i] if n > 1 else axes[0]
        ax.hist(data[col].dropna(), bins=bins, edgecolor='white', alpha=0.8)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel('')

    # Hide unused axes
    if n > 1:
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    
