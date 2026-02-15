import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from .data_prep import equal_freq


# ---------------------------------------------------------------
# cross_plot
# ---------------------------------------------------------------
def cross_plot(data, input, target, auto_binning=True, n_bins=10,
               plot_type='both'):
    """
    Cross-plotting input variable vs. target variable.
    Shows how the input variable is correlated with the target variable,
    getting the likelihood rates for each input's bin/bucket.
    Equivalent to funModeling::cross_plot in R.

    Parameters:
    -----------
    data: data frame source
    input: string input variable name, or list of variable names
    target: string target variable name (binary)
    auto_binning: if True, numeric variables are automatically binned
                  using equal_freq with n_bins. Default True.
    n_bins: number of bins for auto_binning, default 10
    plot_type: 'both' (default), 'percentual', or 'quantity'

    Returns:
    --------
    None (displays plots).

    Example:
    --------
    >> cross_plot(heart_disease, input='gender', target='has_heart_disease')
    >> cross_plot(heart_disease, input=['age', 'chest_pain'], target='has_heart_disease')
    """
    if isinstance(input, list):
        for var in input:
            _cross_plot_logic(data, var, target, auto_binning, n_bins, plot_type)
        return

    _cross_plot_logic(data, input, target, auto_binning, n_bins, plot_type)


def _cross_plot_logic(data, input, target, auto_binning, n_bins, plot_type):
    """Internal function to generate cross_plot for a single variable."""
    df = data[[input, target]].dropna().copy()

    # Auto binning for numeric variables
    if pd.api.types.is_numeric_dtype(df[input]):
        q_unique = df[input].nunique()
        if auto_binning and q_unique > 20:
            df[input] = equal_freq(df[input], n_bins=n_bins)
    else:
        if df[input].nunique() > 50:
            print(f'Skipping "{input}": more than 50 unique values.')
            return

    # Infer the less representative class (commonly the one to predict)
    val_counts = df[target].value_counts()
    pos_class = str(val_counts.idxmin())
    neg_class = str(val_counts.idxmax())

    analysis = df.groupby([input, target]).size().unstack(fill_value=0)
    # Ensure column order: neg_class first, pos_class second
    if neg_class in analysis.columns and pos_class in analysis.columns:
        analysis = analysis[[neg_class, pos_class]]

    pct = analysis.div(analysis.sum(axis=1), axis=0)

    colors = ['#00BFC4', '#F8766D']

    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        _plot_percentual(pct, ax1, input, target, colors)
        _plot_quantity(analysis, ax2, input, target, colors)
        plt.tight_layout()
        plt.show()
    elif plot_type == 'percentual':
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_percentual(pct, ax, input, target, colors)
        plt.tight_layout()
        plt.show()
    elif plot_type == 'quantity':
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_quantity(analysis, ax, input, target, colors)
        plt.tight_layout()
        plt.show()


def _plot_percentual(pct, ax, input_name, target, colors):
    """Internal: stacked percentual bar chart."""
    pct_100 = pct * 100
    pct_100.plot(kind='bar', stacked=True, ax=ax, color=colors,
                 width=0.7, edgecolor='white', linewidth=0.5)
    for i in range(len(pct_100)):
        cumulative = 0
        for j in range(len(pct_100.columns)):
            height = pct_100.iloc[i, j]
            if height > 5:
                ax.text(i, cumulative + height / 2,
                        f'{height:.1f}',
                        ha='center', va='center', fontsize=9, color='black')
            cumulative += height

    ax.set_xlabel(input_name, fontsize=12)
    ax.set_ylabel(f'{target} (%)', fontsize=12)
    ax.set_xticklabels([str(x) for x in pct_100.index], rotation=45, ha='right')
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.legend(title=target, loc='upper right')


def _plot_quantity(analysis, ax, input_name, target, colors):
    """Internal: grouped quantity bar chart."""
    analysis.plot(kind='bar', ax=ax, color=colors,
                  width=0.7, edgecolor='white', linewidth=0.5)
    for container in ax.containers:
        ax.bar_label(container, fontsize=9)

    ax.set_xlabel(input_name, fontsize=12)
    ax.set_ylabel(f'{target} (count)', fontsize=12)
    ax.set_xticklabels([str(x) for x in analysis.index], rotation=45, ha='right')
    ax.legend(title=target, loc='upper right')


# ---------------------------------------------------------------
# var_rank_info
# ---------------------------------------------------------------
def var_rank_info(data, target):
    """
    Variable importance ranking based on Information Theory.
    Computes: entropy (en), mutual information (mi), information gain (ig),
    and gain ratio (gr) for each variable against the target.
    Equivalent to funModeling::var_rank_info in R.

    Parameters:
    -----------
    data: data frame with input variables and target
    target: string, target variable name

    Returns:
    --------
    DataFrame with columns: var, en, mi, ig, gr.
    Sorted by gain ratio descending.

    Example:
    --------
    >> var_rank_info(heart_disease, 'has_heart_disease')
    """
    results = []
    y = data[target]
    input_vars = [c for c in data.columns if c != target]

    def entropy(series):
        probs = series.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs + 1e-10))

    target_entropy = entropy(y)

    for var in input_vars:
        df_temp = data[[var, target]].dropna()
        x = df_temp[var]
        y_temp = df_temp[target]

        # If numeric, discretize for mutual information calculation
        if pd.api.types.is_numeric_dtype(x):
            try:
                x = pd.qcut(x, q=10, duplicates='drop', labels=False)
            except ValueError:
                x = pd.cut(x, bins=10, labels=False)

        # Entropy of the variable
        en = entropy(x)

        # Mutual information (sklearn returns nats, convert to bits)
        mi = mutual_info_score(y_temp.astype(str), x.astype(str))
        mi_bits = mi / np.log(2)

        # Information gain = mutual information in bits
        ig = mi_bits

        # Gain ratio
        gr = ig / en if en > 0 else 0

        results.append({
            'var': var,
            'en': round(en, 4),
            'mi': round(mi_bits, 4),
            'ig': round(ig, 4),
            'gr': round(gr, 4)
        })

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('gr', ascending=False).reset_index(drop=True)
    return df_result


# ---------------------------------------------------------------
# plotar
# ---------------------------------------------------------------
def plotar(data, input, target, plot_type='boxplot'):
    """
    Plot numeric variables grouped by a target variable.
    Generates boxplots or density histograms for each input variable.
    Equivalent to funModeling::plotar in R.

    Parameters:
    -----------
    data: data frame
    input: string or list of strings with numeric variable names
    target: string, target variable name (categorical)
    plot_type: 'boxplot' (default) or 'histdens'

    Returns:
    --------
    None (displays plots).

    Example:
    --------
    >> plotar(heart_disease, input='age', target='has_heart_disease')
    >> plotar(heart_disease, input=['age', 'oldpeak'], target='has_heart_disease', plot_type='histdens')
    """
    if isinstance(input, str):
        input = [input]

    for var in input:
        if plot_type == 'boxplot':
            fig, ax = plt.subplots(figsize=(5, 3))
            data.boxplot(column=var, by=target, ax=ax,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightblue'))
            ax.set_title(f'{var}')
            ax.set_xlabel(target)
            ax.set_ylabel(var)
            plt.suptitle('')
            plt.tight_layout()
            plt.show()

        elif plot_type == 'histdens':
            fig, ax = plt.subplots(figsize=(5, 3))
            for val in sorted(data[target].dropna().unique()):
                subset = data[data[target] == val][var].dropna()
                subset.plot.kde(ax=ax, label=str(val))
                ax.axvline(subset.mean(), linestyle='--', alpha=0.5)
            ax.set_xlabel(var)
            ax.set_ylabel('Densidad')
            ax.set_title(f'{var}')
            ax.legend(title=target)
            plt.tight_layout()
            plt.show()
