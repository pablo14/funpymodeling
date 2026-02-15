import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def todf(data):
    """
    It converts almost any object to pandas dataframe. It supports: 1D/2D list, 1D/2D arrays, pandas series. If the object containts +2D it returns an error.
    Parameters:
    -----------
    data: data
    
    Returns:
    --------
    A pandas dataframe.

    Example:
    --------
    >> from numpy import array

    # Different case study:
    >> list_1d = [11, 12, 5, 2] 
    >> todf(list_1d)
    >> list_2d = [[11, 12, 5, 2], [15,24, 6,10], [10, 8, 12, 5], [12,15,8,6]]
    >> todf(list_2d)
    >> list_3d = [[[11, 12, 5, 2], [15,24, 6,10], [10, 8, 12, 5], [12,15,8,6]]]
    >> todf(list_3d)
    >> array_1d = array(list_1d)
    >> todf(array_1d)
    >> array_2d = array(list_2d)
    >> todf(array_2d)
    >> pd_df=pd.DataFrame({'v1':[11, 12, 5, 2], 'v2':[15,24, 6,10]}) # ok
    >> todf(pd_df)
    >> pd_series=pd_df.v1
    """
    if isinstance(data, list):
        data=np.array(data)

    if(len(data.shape))>2:
        raise Exception("I live in flattland! (can't handle objects with more than 2 dimensions)") 

    if isinstance(data, pd.Series):
        data2=pd.DataFrame({data.name: data})
    elif isinstance(data, np.ndarray):
        if(data.shape==1):
            data2=pd.DataFrame({'var': data}).convert_dtypes()
        else:
            data2=pd.DataFrame(data).convert_dtypes()
    else: 
        data2=data
        
    return data2


# ---------------------------------------------------------------
# equal_freq
# ---------------------------------------------------------------
def equal_freq(var, n_bins=5):
    """
    Equal frequency binning. Splits the variable into n_bins segments
    of approximately equal size. Wrapper of pandas qcut.
    Equivalent to funModeling::equal_freq in R (which wraps Hmisc::cut2).

    Parameters:
    -----------
    var: numeric vector (pandas Series, list, or numpy array)
    n_bins: number of bins to split 'var' by equal frequency

    Returns:
    --------
    The binned variable as a Categorical Series.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> equal_freq(iris['sepal_length'], n_bins=5)
    """
    return pd.qcut(var, q=n_bins, duplicates='drop')


# ---------------------------------------------------------------
# tukey_outlier
# ---------------------------------------------------------------
def tukey_outlier(x):
    """
    Tukey outlier thresholds based on IQR * 3.
    Returns a dict with 'lower' and 'upper' boundaries.
    Equivalent to funModeling::tukey_outlier in R.

    Parameters:
    -----------
    x: numeric vector

    Returns:
    --------
    Dict with keys 'lower' and 'upper'.

    Example:
    --------
    >> tukey_outlier([1, 2, 3, 4, 5, 100])
    """
    x = pd.Series(x).dropna()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = round(q1 - 3 * iqr, 2)
    upper = round(q3 + 3 * iqr, 2)
    return {'lower': lower, 'upper': upper}


# ---------------------------------------------------------------
# hampel_outlier
# ---------------------------------------------------------------
def hampel_outlier(x, k_mad_value=3):
    """
    Hampel outlier thresholds based on median +/- k * MAD.
    Returns a dict with 'lower' and 'upper' boundaries.
    Equivalent to funModeling::hampel_outlier in R.

    Parameters:
    -----------
    x: numeric vector
    k_mad_value: multiplier for MAD, default=3

    Returns:
    --------
    Dict with keys 'lower' and 'upper'.

    Example:
    --------
    >> hampel_outlier([1, 2, 3, 4, 5, 100])
    """
    x = pd.Series(x).dropna()
    med = x.median()
    mad = median_abs_deviation(x, scale='normal')
    lower = med - k_mad_value * mad
    upper = med + k_mad_value * mad
    return {'lower': round(lower, 2), 'upper': round(upper, 2)}


# ---------------------------------------------------------------
# prep_outliers
# ---------------------------------------------------------------
def prep_outliers(data, input=None, type='stop', method='tukey',
                  top_percent=0.01, bottom_percent=0.01, k_mad_value=3):
    """
    Handle outliers using tukey, hampel, or bottom_top method.
    type='set_na': converts outliers to NaN (recommended for statistical analysis).
    type='stop': clips values at the threshold (recommended for predictive modeling).
    Equivalent to funModeling::prep_outliers in R.

    Parameters:
    -----------
    data: data frame or single numeric vector/Series
    input: list of column names to process. If None and data is a DataFrame,
           runs for all numeric columns.
    type: 'stop' or 'set_na'
    method: 'tukey', 'hampel', or 'bottom_top'
    top_percent: upper percentile to clip (only for method='bottom_top')
    bottom_percent: lower percentile to clip (only for method='bottom_top')
    k_mad_value: multiplier for MAD (only for method='hampel')

    Returns:
    --------
    Transformed data frame or Series.

    Example:
    --------
    >> import pandas as pd
    >> df = pd.DataFrame({'v1': range(100)})
    >> prep_outliers(df, input=['v1'], type='stop', method='tukey')
    """
    if isinstance(data, pd.Series):
        data = data.to_frame(name='value')
        input = ['value']
        single = True
    else:
        data = data.copy()
        single = False
        if input is None:
            input = data.select_dtypes(include=[np.number]).columns.tolist()

    for col in input:
        x = data[col]
        if method == 'tukey':
            bounds = tukey_outlier(x)
        elif method == 'hampel':
            bounds = hampel_outlier(x, k_mad_value)
        elif method == 'bottom_top':
            lower_val = x.quantile(bottom_percent)
            upper_val = x.quantile(1 - top_percent)
            bounds = {'lower': lower_val, 'upper': upper_val}
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tukey', 'hampel', or 'bottom_top'.")

        if type == 'set_na':
            data[col] = x.where(
                (x >= bounds['lower']) & (x <= bounds['upper']),
                other=np.nan)
        elif type == 'stop':
            data[col] = x.clip(lower=bounds['lower'], upper=bounds['upper'])
        else:
            raise ValueError(f"Unknown type: {type}. Use 'stop' or 'set_na'.")

    if single:
        return data['value']
    return data


# ---------------------------------------------------------------
# categ_analysis
# ---------------------------------------------------------------
def categ_analysis(data, input, target):
    """
    Analyze the relationship between a categorical variable and a binary target.
    The positive class is inferred as the less representative one.
    Equivalent to funModeling::categ_analysis in R.

    Parameters:
    -----------
    data: data frame
    input: string, categorical input variable name (or list of names)
    target: string, binary target variable name

    Returns:
    --------
    DataFrame with columns: input, mean_target, sum_target, perc_target,
    q_rows, perc_rows. Sorted by mean_target descending.

    Example:
    --------
    >> categ_analysis(heart_disease, 'gender', 'has_heart_disease')
    """
    if isinstance(input, list):
        for col in input:
            res = categ_analysis(data, col, target)
            print(res)
            print()
        return

    df = data[[input, target]].copy()
    val_counts = df[target].value_counts()
    minority_class = val_counts.idxmin()
    df['target_num'] = (df[target] == minority_class).astype(int)

    grp = df.groupby(input).agg(
        mean_target=('target_num', 'mean'),
        sum_target=('target_num', 'sum'),
        q_rows=('target_num', 'count')
    ).reset_index()

    total_target = grp['sum_target'].sum()
    total_rows = grp['q_rows'].sum()
    grp['perc_target'] = round(grp['sum_target'] / total_target, 3) if total_target > 0 else 0
    grp['perc_rows'] = round(grp['q_rows'] / total_rows, 3)
    grp['mean_target'] = round(grp['mean_target'], 3)

    grp = grp.sort_values('mean_target', ascending=False).reset_index(drop=True)
    col_order = [input, 'mean_target', 'sum_target', 'perc_target', 'q_rows', 'perc_rows']
    return grp[col_order]


# ---------------------------------------------------------------
# auto_grouping
# ---------------------------------------------------------------
def auto_grouping(data, input, target, n_groups, model='kmeans', seed=999):
    """
    Reduce cardinality of a categorical variable using clustering
    on the profiling metrics (perc_rows, perc_target).
    Equivalent to funModeling::auto_grouping in R.

    Parameters:
    -----------
    data: data frame
    input: string, categorical input variable name
    target: string, binary target variable name
    n_groups: int, number of groups for the new category
    model: 'kmeans' (default). Only kmeans is supported.
    seed: random seed for reproducibility

    Returns:
    --------
    Dict with keys:
      - 'recateg_results': profiling of the new groups
      - 'df_equivalence': mapping from original to new categories
      - 'fit_cluster': the fitted KMeans model

    Example:
    --------
    >> auto_grouping(data_country, 'country', 'has_flu', n_groups=8)
    """
    prof = categ_analysis(data, input, target)
    features = prof[['perc_rows', 'perc_target']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_groups, random_state=seed, n_init=10)
    km.fit(features_scaled)
    prof['cluster'] = km.labels_
    prof[f'{input}_rec'] = 'group_' + (prof['cluster'] + 1).astype(str)

    df_equivalence = prof[[input, f'{input}_rec']].copy()

    data_merged = data.merge(df_equivalence, on=input, how='inner')
    recateg = categ_analysis(data_merged, f'{input}_rec', target)

    return {
        'recateg_results': recateg,
        'df_equivalence': df_equivalence,
        'fit_cluster': km
    }


# ---------------------------------------------------------------
# discretize_get_bins
# ---------------------------------------------------------------
def discretize_get_bins(data, input, n_bins=5):
    """
    Get bin thresholds for equal-frequency discretization.
    Equivalent to funModeling::discretize_get_bins in R.

    Parameters:
    -----------
    data: data frame
    input: list of column names to discretize
    n_bins: number of bins/segments for each variable

    Returns:
    --------
    Dict mapping column name -> array of bin edges.

    Example:
    --------
    >> d_bins = discretize_get_bins(heart_disease, ['age', 'oldpeak'], n_bins=5)
    """
    result = {}
    for col in input:
        x = data[col].dropna()
        _, bins = pd.qcut(x, q=n_bins, retbins=True, duplicates='drop')
        bins[0] = -np.inf
        bins[-1] = np.inf
        result[col] = bins
    return result


# ---------------------------------------------------------------
# discretize_df
# ---------------------------------------------------------------
def discretize_df(data, data_bins):
    """
    Apply discretization bins to a data frame. Missing values are
    assigned the category 'NA.'.
    Equivalent to funModeling::discretize_df in R.

    Parameters:
    -----------
    data: data frame to discretize
    data_bins: dict from discretize_get_bins, mapping column -> bin edges

    Returns:
    --------
    Data frame with the discretized variables.

    Example:
    --------
    >> d_bins = discretize_get_bins(heart_disease, ['age', 'oldpeak'], n_bins=5)
    >> heart_disease_disc = discretize_df(heart_disease, d_bins)
    """
    df = data.copy()
    for col, bins in data_bins.items():
        labels = []
        for i in range(len(bins) - 1):
            if bins[i] == -np.inf:
                labels.append(f'[-Inf, {bins[i+1]:.1f})')
            elif bins[i+1] == np.inf:
                labels.append(f'[{bins[i]:.1f}, Inf]')
            else:
                labels.append(f'[{bins[i]:.1f}, {bins[i+1]:.1f})')
        na_mask = df[col].isna()
        df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        df[col] = df[col].cat.add_categories('NA.')
        df.loc[na_mask, col] = 'NA.'
    return df

