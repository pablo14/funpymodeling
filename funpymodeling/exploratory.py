import pandas as pd
import numpy as np
from .data_prep import todf

def status(data):
    """
    For each variable it returns: Quantity and percentage of zeros (q_zeros and p_zeros respectevly). Same metrics for NA values (q_NA/p_na), and infinite values (q_inf/p_inf). Last two columns indicates data type and quantity of unique values.
    status can be used for EDA or in a data flow to spot errors or take actions based on the result.
    
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
    >> # dataframe as input
    >> status(iris)
    >> # single columns:
    >> status(iris['species'])
    """
    data2=todf(data)

    # total de rows
    tot_rows=len(data2)
    
    # total de nan
    d2=data2.isnull().sum().reset_index()
    d2.columns=['variable', 'q_nan']
    
    # percentage of nan
    d2[['p_nan']]=d2[['q_nan']]/tot_rows
    
    # num of zeros
    d2['q_zeros']=(data2==0).sum().values

    # perc of zeros
    d2['p_zeros']=d2[['q_zeros']]/tot_rows

    # total unique values
    d2['unique']=data2.nunique().values
    
    # get data types per column
    d2['type']=[str(x) for x in data2.dtypes.values]
    
    return(d2)


def corr_pair(data, method='pearson'):
    """
    Calcuate the correlations among all numeric features. Non-numeric are excluded since it uses the `corr` pandas function.
    It's useful to quickly extract those correlated input features and the correlation between the input and the target variable.
    
    Parameters:
    -----------
    data: pandas data containing the variables to calculate the correlation
    method: `pearson` as default, same as `corr` function in pandas. 
    Returns:
    --------
    A pandas dataframe containing pairwaise correlation, R and R2 statistcs

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> corr_pair(iris)
    """
    data2=todf(data)
    
    d_cor=data2.corr(method)

    d_cor2=d_cor.reset_index() # generates index as column

    d_long=d_cor2.melt(id_vars='index') # to long format, each row 1 var

    d_long.columns=['v1', 'v2', 'R']
    
    d_long[['R2']]=d_long[['R']]**2
    
    d_long2=d_long.query("v1 != v2") # don't need the auto-correlation

    return(d_long2)


def num_vars(data, exclude_var=None):
    """
    Returns the numeric variable names. Useful to use with pipelines or any other method in which we need to keep numeric variables. It `exclude_var` can be a list with the variable names to skip in the result. Useful when we want to skip the target variable (i.e. in a data transformation).
    It's also available for categorical variables in the function `cat_vars()`
    Parameters:
    -----------
    data: pandas dataframe
    exclude_var: list of variable names to exclude from the result
    
    Returns:
    --------
    A list with all the numeric variable names.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> num_vars(iris)
    """
    num_v = data.select_dtypes(include=['int64', 'float64']).columns
    if exclude_var is not None: 
        num_v=num_v.drop(exclude_var)
    return num_v

def cat_vars(data, exclude_var=None):
    """
    Returns the categoric variable names. Useful to use with pipelines or any other method in which we need to keep categorical variables. It `exclude_var` can be a list with the variable names to skip in the result. Useful when we want to skip the target variable (i.e. in a data transformation). It will include all `object`, `category` and `string` variables.
    It's also available for numeric variables in the function `num_vars()`
    
    Parameters:
    -----------
    data: pandas dataframe
    exclude_var: list of variable names to exclude from the result
    
    Returns:
    --------
    A list with all the categoric variable names.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> cat_vars(iris)
    """
    cat_v = data.select_dtypes(include=['object','category', 'string']).columns
    if exclude_var is not None: 
        cat_v=cat_v.drop(exclude_var)
    return cat_v


def profiling_num(data):
    """
    Get a metric table with many indicators for all numerical variables, automatically skipping the non-numerical variables. Current metrics are: mean, std_dev: standard deviation, all the p_XX: percentile at XX number, skewness, kurtosis, iqr: inter quartile range, variation_coef: the ratio of sd/mean, range_98 is the limit for which the 98% of fall, range_80 similar to range_98 but with 80%. All NA values will be skipped from calculations.

    Parameters:
    -----------
    data: pandas  series/dataframe, numpy 1D/2D array
    
    Returns:
    --------
    A dataframe in which each row is an input variable, and each column an statistic.

    Example:
    --------
    >> import seaborn as sns
    >> iris = sns.load_dataset('iris')
    >> profiling_num(iris)
    """
    
    # handling different inputs to dataframe
    data=todf(data)
    
    # explicit keep the num vars
    d=data[num_vars(data)]
    
    des1=pd.DataFrame({'mean':d.mean().transpose(), 
                   'std_dev':d.std().transpose()})

    des1['variation_coef']=des1['std_dev']/des1['mean']
    
    d_quant=d.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose().add_prefix('p_')
    
    des2=des1.join(d_quant, how='outer')
    
    des_final=des2.copy()
    
    des_final['variable'] = des_final.index
    
    des_final=des_final.reset_index(drop=True)
    
    des_final=des_final[['variable', 'mean', 'std_dev','variation_coef', 'p_0.01', 'p_0.05', 'p_0.25', 'p_0.5', 'p_0.75', 'p_0.95', 'p_0.99']]
    
    return des_final



def _freq_tbl_logic(var, name):
    """
    For internal use. Related to `freq_tbl`.

    Parameters:
    -----------
    var: pandas series
    name: column name (string)
    
    Returns:
    --------
    Dataframe with the metrics

    Example:
    --------

    """
    cnt=var.value_counts()
    df_res=pd.DataFrame({'frequency': var.value_counts(), 'percentage': var.value_counts()/len(var)})
    df_res.reset_index(drop=True)
    
    df_res[name] = df_res.index
    
    df_res=df_res.reset_index(drop=True)
    
    df_res['cumulative_perc'] = df_res.percentage.cumsum()/df_res.percentage.sum()
    
    df_res=df_res[[name, 'frequency', 'percentage', 'cumulative_perc']]
    
    return df_res



def freq_tbl(data):
    """
    Frequency table for categorical variables. It retrieves the frequency, perrcentage and cummulative percentage for each categorical variables (excluding the numerical ones).

    Parameters:
    -----------
    data: pandas series/dataframe, numpy 1D/2D array
    
    Returns:
    --------
    If a single variable is passed, then it returns the table with the results (useful to be used in a processes and take actions based on the result.).
    If it contains more than one varible, it will print in the console the result for all the categorical variables (based on cat_vars). 

    Example:
    --------
    > import seaborn as sns
    > tips=sns.load_dataset('tips')
    > freq_tbl(tips)
    """
    data=todf(data)
    
    cat_v=cat_vars(data)
    if(len(cat_v)==0):
        return('No categorical variables to analyze.')
    
    if(len(cat_v)>1):
        for col in cat_v:
            print(_freq_tbl_logic(data[col], name=col))
            print('\n----------------------------------------------------------------\n')
    else:
        # if only 1 column, then return the table for that variable
        col=cat_v[0]
        return _freq_tbl_logic(data[col], name=col)
    
