"""
Tests for prep_outliers, tukey_outlier, hampel_outlier.
Based on R funModeling myTESTS/test_prep_outliers.R
"""
import pytest
import pandas as pd
import numpy as np

from funpymodeling import prep_outliers, tukey_outlier, hampel_outlier


# ---------------------------------------------------------------------------
# Fixture: simple dataframe
# ---------------------------------------------------------------------------
@pytest.fixture
def df_outliers():
    np.random.seed(42)
    v1 = list(range(100))
    v2 = list(np.random.normal(100, 10, 100))
    return pd.DataFrame({'v1': v1, 'v2': v2})


# ---------------------------------------------------------------------------
# method="bottom_top" + 1 var, stop
# ---------------------------------------------------------------------------
def test_bottom_top_1var_stop(df_outliers):
    result = prep_outliers(
        df_outliers, input=['v1'], type='stop', method='bottom_top',
        top_percent=0.01, bottom_percent=0.01
    )
    assert isinstance(result, pd.DataFrame)
    # Values should be clipped
    q01 = df_outliers['v1'].quantile(0.01)
    q99 = df_outliers['v1'].quantile(0.99)
    assert result['v1'].min() >= q01
    assert result['v1'].max() <= q99


# ---------------------------------------------------------------------------
# passing a vector (Series) returning a vector
# ---------------------------------------------------------------------------
def test_bottom_top_vector():
    s = pd.Series(range(100))
    result = prep_outliers(s, type='stop', method='bottom_top',
                           top_percent=0.01, bottom_percent=0.01)
    assert isinstance(result, pd.Series)
    assert len(result) == 100


# ---------------------------------------------------------------------------
# 2 var, set_na
# ---------------------------------------------------------------------------
def test_bottom_top_2var_set_na(df_outliers):
    result = prep_outliers(
        df_outliers, input=['v1', 'v2'], type='set_na', method='bottom_top',
        top_percent=0.01, bottom_percent=0.01
    )
    assert isinstance(result, pd.DataFrame)
    # Some values should be NaN now (the outliers)
    # Not necessarily, depends on data, but structure should be correct
    assert 'v1' in result.columns
    assert 'v2' in result.columns


# ---------------------------------------------------------------------------
# method="tukey" — 1 var, stop
# ---------------------------------------------------------------------------
def test_tukey_1var_stop(df_outliers):
    result = prep_outliers(df_outliers, input=['v1'], type='stop', method='tukey')
    assert isinstance(result, pd.DataFrame)
    bounds = tukey_outlier(df_outliers['v1'])
    assert result['v1'].min() >= bounds['lower']
    assert result['v1'].max() <= bounds['upper']


# ---------------------------------------------------------------------------
# method="tukey" — set_na
# ---------------------------------------------------------------------------
def test_tukey_set_na(df_outliers):
    result = prep_outliers(df_outliers, input=['v1'], type='set_na', method='tukey')
    bounds = tukey_outlier(df_outliers['v1'])
    # Values outside bounds should be NaN
    original_outside = (
        (df_outliers['v1'] < bounds['lower']) | (df_outliers['v1'] > bounds['upper'])
    ).sum()
    result_na = result['v1'].isna().sum()
    assert result_na == original_outside


# ---------------------------------------------------------------------------
# method="tukey" forcing outliers
# ---------------------------------------------------------------------------
def test_tukey_with_forced_outliers():
    data = pd.DataFrame({'v1': list(range(50)) + [500, -200]})
    result = prep_outliers(data, input=['v1'], type='stop', method='tukey')
    bounds = tukey_outlier(data['v1'])
    assert result['v1'].max() <= bounds['upper']
    assert result['v1'].min() >= bounds['lower']


# ---------------------------------------------------------------------------
# method="hampel"
# ---------------------------------------------------------------------------
def test_hampel_stop(df_outliers):
    result = prep_outliers(df_outliers, input=['v1'], type='stop', method='hampel')
    bounds = hampel_outlier(df_outliers['v1'])
    assert result['v1'].min() >= bounds['lower']
    assert result['v1'].max() <= bounds['upper']


def test_hampel_set_na(df_outliers):
    result = prep_outliers(df_outliers, input=['v1'], type='set_na', method='hampel')
    bounds = hampel_outlier(df_outliers['v1'])
    original_outside = (
        (df_outliers['v1'] < bounds['lower']) | (df_outliers['v1'] > bounds['upper'])
    ).sum()
    result_na = result['v1'].isna().sum()
    assert result_na == original_outside


# ---------------------------------------------------------------------------
# method="hampel" forcing outliers
# ---------------------------------------------------------------------------
def test_hampel_with_forced_outliers():
    data = pd.DataFrame({'v1': list(range(50)) + [500, -200]})
    result_stop = prep_outliers(data, input=['v1'], type='stop', method='hampel')
    bounds = hampel_outlier(data['v1'])
    assert result_stop['v1'].max() <= bounds['upper']
    assert result_stop['v1'].min() >= bounds['lower']


# ---------------------------------------------------------------------------
# tukey_outlier directly
# ---------------------------------------------------------------------------
def test_tukey_outlier_basic():
    bounds = tukey_outlier([1, 2, 3, 4, 5, 100])
    assert 'lower' in bounds
    assert 'upper' in bounds
    assert bounds['lower'] < bounds['upper']


# ---------------------------------------------------------------------------
# hampel_outlier directly
# ---------------------------------------------------------------------------
def test_hampel_outlier_basic():
    bounds = hampel_outlier([1, 2, 3, 4, 5, 100])
    assert 'lower' in bounds
    assert 'upper' in bounds
    assert bounds['lower'] < bounds['upper']


# ---------------------------------------------------------------------------
# No input specified → all numeric columns
# ---------------------------------------------------------------------------
def test_prep_outliers_no_input(df_outliers):
    result = prep_outliers(df_outliers, type='stop', method='tukey')
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df_outliers.shape


# ---------------------------------------------------------------------------
# Invalid method / type
# ---------------------------------------------------------------------------
def test_invalid_method(df_outliers):
    with pytest.raises(ValueError, match="Unknown method"):
        prep_outliers(df_outliers, input=['v1'], type='stop', method='invalid')


def test_invalid_type(df_outliers):
    with pytest.raises(ValueError, match="Unknown type"):
        prep_outliers(df_outliers, input=['v1'], type='invalid', method='tukey')


# ---------------------------------------------------------------------------
# only bottom — bottom_top with top_percent=0
# ---------------------------------------------------------------------------
def test_bottom_top_only_bottom():
    data = pd.DataFrame({'v1': list(range(100))})
    result = prep_outliers(
        data, input=['v1'], type='stop', method='bottom_top',
        top_percent=0, bottom_percent=0.05
    )
    q05 = data['v1'].quantile(0.05)
    assert result['v1'].min() >= q05


# ---------------------------------------------------------------------------
# Skewed variable
# ---------------------------------------------------------------------------
def test_skewed_variable():
    np.random.seed(42)
    skewed = pd.DataFrame({'v1': np.random.exponential(scale=10, size=200)})
    result = prep_outliers(skewed, input=['v1'], type='stop', method='tukey')
    bounds = tukey_outlier(skewed['v1'])
    assert result['v1'].max() <= bounds['upper']
    assert result['v1'].min() >= bounds['lower']
