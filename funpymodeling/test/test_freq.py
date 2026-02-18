"""
Tests for freq_tbl and freq_plot, based on R funModeling myTESTS/test_freq.R
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for testing

from funpymodeling import freq_tbl, freq_plot, cat_vars


# ---------------------------------------------------------------------------
# Helper: minimal heart_disease-like dataframe
# ---------------------------------------------------------------------------
@pytest.fixture
def heart_disease():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n),
        'thal': np.random.choice(['normal', 'fixed_defect', 'reversable_defect', np.nan], n),
        'chest_pain': np.random.choice(['typ_angina', 'asympt', 'non_anginal', 'abn'], n),
        'age': np.random.randint(30, 80, n),
        'max_heart_rate': np.random.randint(100, 200, n),
        'has_heart_disease': np.random.choice(['yes', 'no'], n),
    })


# ---------------------------------------------------------------------------
# R test: freq(heart_disease$gender) — 1 variable
# ---------------------------------------------------------------------------
def test_freq_tbl_single_var(heart_disease):
    result = freq_tbl(heart_disease, input='gender')
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['gender', 'frequency', 'percentage', 'cumulative_perc']
    assert result['frequency'].sum() == len(heart_disease)
    assert result['cumulative_perc'].iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# R test: freq(heart_disease, c("gender","thal")) — 2 variables
# ---------------------------------------------------------------------------
def test_freq_tbl_two_vars(heart_disease, capsys):
    result = freq_tbl(heart_disease, input=['gender', 'thal'])
    captured = capsys.readouterr()
    # When multiple vars, it prints each table and returns a summary string
    assert 'Variables processed' in result
    assert 'gender' in captured.out
    assert 'thal' in captured.out


# ---------------------------------------------------------------------------
# R test: freq(heart_disease) — no vars specified → all categorical
# ---------------------------------------------------------------------------
def test_freq_tbl_all_categorical(heart_disease, capsys):
    result = freq_tbl(heart_disease)
    captured = capsys.readouterr()
    cat_cols = list(cat_vars(heart_disease))
    assert len(cat_cols) > 0
    assert 'Variables processed' in result


# ---------------------------------------------------------------------------
# R test: a=as.factor(1:300); b=freq(a)  — high cardinality
# ---------------------------------------------------------------------------
def test_freq_tbl_high_cardinality():
    high_card = pd.DataFrame({'x': [str(i) for i in range(300)]})
    result = freq_tbl(high_card, input='x')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 300
    assert result['frequency'].sum() == 300


# ---------------------------------------------------------------------------
# R test: factor var — freq(tt$b) where b=factor(c("aa","vv","vv"))
# ---------------------------------------------------------------------------
def test_freq_tbl_factor_var():
    b = pd.Categorical(['aa', 'vv', 'vv'])
    df = pd.DataFrame({'b': b})
    result = freq_tbl(df, input='b')
    assert isinstance(result, pd.DataFrame)
    assert result.loc[result['b'] == 'vv', 'frequency'].values[0] == 2
    assert result.loc[result['b'] == 'aa', 'frequency'].values[0] == 1


# ---------------------------------------------------------------------------
# R test: freq(a) where a=c(NA,NA,NA) — all NA
# ---------------------------------------------------------------------------
def test_freq_tbl_all_na():
    df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
    result = freq_tbl(df, input='a')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result['a'].iloc[0] == 'NA'
    assert result['frequency'].iloc[0] == 3


# ---------------------------------------------------------------------------
# R test: freq(mtcars) — all numerical → no categorical vars message
# ---------------------------------------------------------------------------
def test_freq_tbl_all_numerical():
    mtcars = pd.DataFrame({
        'mpg': [21.0, 21.0, 22.8],
        'cyl': [6, 6, 4],
        'disp': [160.0, 160.0, 108.0],
    })
    result = freq_tbl(mtcars)
    assert result == 'No categorical variables to analyze.'


# ---------------------------------------------------------------------------
# Bug fix test: categorical dtype with NaN should not crash
# ---------------------------------------------------------------------------
def test_freq_tbl_categorical_with_nan():
    s = pd.Series(pd.Categorical(['a', 'b', 'a', None, 'b', 'b']))
    df = pd.DataFrame({'cat_col': s})
    result = freq_tbl(df, input='cat_col')
    assert isinstance(result, pd.DataFrame)
    assert 'NA' in result['cat_col'].values
    assert result['frequency'].sum() == 6


def test_freq_tbl_categorical_without_nan():
    s = pd.Series(pd.Categorical(['a', 'b', 'a', 'b', 'b']))
    df = pd.DataFrame({'cat_col': s})
    result = freq_tbl(df, input='cat_col')
    assert isinstance(result, pd.DataFrame)
    assert result['frequency'].sum() == 5
    assert 'NA' not in result['cat_col'].values


# ---------------------------------------------------------------------------
# na_rm parameter test
# ---------------------------------------------------------------------------
def test_freq_tbl_na_rm():
    df = pd.DataFrame({'x': ['a', 'b', 'a', np.nan, 'b', 'b']})
    result_with_na = freq_tbl(df, input='x', na_rm=False)
    result_without_na = freq_tbl(df, input='x', na_rm=True)
    assert result_with_na['frequency'].sum() == 6
    assert result_without_na['frequency'].sum() == 5


# ---------------------------------------------------------------------------
# Mixed dtypes: factor var in a dataframe with NAs in other columns
# R test: tt=data.frame(a,b) where a=c(NA,NA,NA), b=factor(c("aa","vv","vv"))
# ---------------------------------------------------------------------------
def test_freq_tbl_mixed_na_and_factor():
    a = [np.nan, np.nan, np.nan]
    b = pd.Categorical(['aa', 'vv', 'vv'])
    tt = pd.DataFrame({'a': a, 'b': b})
    # freq on the factor column
    result = freq_tbl(tt, input='b')
    assert isinstance(result, pd.DataFrame)
    assert result['frequency'].sum() == 3


# ---------------------------------------------------------------------------
# freq_plot smoke tests (just ensure no errors)
# ---------------------------------------------------------------------------
def test_freq_plot_single_var(heart_disease):
    freq_plot(heart_disease, input='gender')


def test_freq_plot_multiple_vars(heart_disease):
    freq_plot(heart_disease, input=['gender', 'chest_pain'])


def test_freq_plot_all_categorical(heart_disease):
    freq_plot(heart_disease)


def test_freq_plot_categorical_dtype_with_nan():
    """Ensure freq_plot also works with categorical dtype containing NaN."""
    s = pd.Series(pd.Categorical(['a', 'b', 'a', None, 'b', 'b']))
    df = pd.DataFrame({'cat_col': s})
    freq_plot(df, input='cat_col')
