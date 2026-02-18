"""
Tests for cross_plot, based on R funModeling myTESTS/tests_cross_plot.R
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from funpymodeling import cross_plot, prep_outliers, equal_freq


# ---------------------------------------------------------------------------
# Fixture: heart_disease-like dataframe
# ---------------------------------------------------------------------------
@pytest.fixture
def heart_disease():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n),
        'chest_pain': np.random.choice(['typ_angina', 'asympt', 'non_anginal', 'abn'], n),
        'age': np.random.randint(30, 80, n),
        'max_heart_rate': np.random.randint(100, 200, n),
        'oldpeak': np.random.uniform(0, 5, n),
        'has_heart_disease': np.random.choice(['yes', 'no'], n),
    })


@pytest.fixture
def mtcars():
    np.random.seed(42)
    n = 32
    return pd.DataFrame({
        'mpg': np.random.uniform(10, 35, n),
        'hp': np.random.uniform(50, 335, n),
        'drat': np.random.uniform(2.5, 5.0, n),
        'vs': np.random.choice([0, 1], n),
    })


# ---------------------------------------------------------------------------
# R test: cross_plot normal — categorical input
# ---------------------------------------------------------------------------
def test_cross_plot_normal(heart_disease):
    cross_plot(data=heart_disease, input='chest_pain', target='has_heart_disease')


# ---------------------------------------------------------------------------
# R test: cross_plot with auto_binning=True — numeric input
# ---------------------------------------------------------------------------
def test_cross_plot_auto_binning(mtcars):
    cross_plot(data=mtcars, input='hp', target='vs', auto_binning=True)


# ---------------------------------------------------------------------------
# R test: cross_plot with auto_binning=False
# ---------------------------------------------------------------------------
def test_cross_plot_no_auto_binning(mtcars):
    cross_plot(data=mtcars, input='hp', target='vs', auto_binning=False)


# ---------------------------------------------------------------------------
# R test: cross_plot with equal_freq binned input
# ---------------------------------------------------------------------------
def test_cross_plot_pre_binned(mtcars):
    mtcars['hp_2'] = equal_freq(mtcars['hp'], 5).astype(str)
    cross_plot(data=mtcars, input='hp_2', target='vs', auto_binning=True)


# ---------------------------------------------------------------------------
# R test: uniq>20 — automatic auto binning
# ---------------------------------------------------------------------------
def test_cross_plot_high_unique_auto(mtcars):
    cross_plot(data=mtcars, input='drat', target='vs')


# ---------------------------------------------------------------------------
# R test: uniq>20 — forcing not binning
# ---------------------------------------------------------------------------
def test_cross_plot_high_unique_no_bin(mtcars):
    cross_plot(data=mtcars, input='drat', target='vs', auto_binning=False)


# ---------------------------------------------------------------------------
# R test: forcing NA in target
# ---------------------------------------------------------------------------
def test_cross_plot_na_in_target(heart_disease):
    heart_disease.loc[0, 'has_heart_disease'] = np.nan
    cross_plot(data=heart_disease, input='chest_pain', target='has_heart_disease')


# ---------------------------------------------------------------------------
# R test: target as numeric
# ---------------------------------------------------------------------------
def test_cross_plot_numeric_target(heart_disease):
    heart_disease['has_heart_disease_num'] = (
        heart_disease['has_heart_disease'] == 'yes'
    ).astype(int)
    cross_plot(data=heart_disease, input='chest_pain', target='has_heart_disease_num')


# ---------------------------------------------------------------------------
# R test: input missing, run for every variable
# ---------------------------------------------------------------------------
def test_cross_plot_all_vars(heart_disease):
    # When input is a list of all columns except target
    inputs = [c for c in heart_disease.columns if c != 'has_heart_disease']
    cross_plot(data=heart_disease, input=inputs, target='has_heart_disease')


# ---------------------------------------------------------------------------
# R test: multiple inputs
# ---------------------------------------------------------------------------
def test_cross_plot_multiple_inputs(heart_disease):
    cross_plot(
        data=heart_disease,
        input=['chest_pain', 'gender'],
        target='has_heart_disease'
    )


# ---------------------------------------------------------------------------
# plot_type options
# ---------------------------------------------------------------------------
def test_cross_plot_percentual(heart_disease):
    cross_plot(data=heart_disease, input='chest_pain',
               target='has_heart_disease', plot_type='percentual')


def test_cross_plot_quantity(heart_disease):
    cross_plot(data=heart_disease, input='chest_pain',
               target='has_heart_disease', plot_type='quantity')
