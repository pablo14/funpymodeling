"""
Tests for plotar, based on R funModeling myTESTS/tests_plotar.R
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from funpymodeling import plotar


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def heart_disease():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n),
        'age': np.random.randint(30, 80, n),
        'max_heart_rate': np.random.randint(100, 200, n),
        'oldpeak': np.random.uniform(0, 5, n),
        'num_vessels_flour': np.random.choice([0, 0, 0, 1, 2, 3], n),
        'has_heart_disease': np.random.choice(['yes', 'no'], n),
    })


# ---------------------------------------------------------------------------
# R test: 1 var, histdens
# ---------------------------------------------------------------------------
def test_plotar_1var_histdens(heart_disease):
    plotar(data=heart_disease, input='age',
           target='has_heart_disease', plot_type='histdens')


# ---------------------------------------------------------------------------
# R test: 1 var, boxplot
# ---------------------------------------------------------------------------
def test_plotar_1var_boxplot(heart_disease):
    plotar(data=heart_disease, input='age',
           target='has_heart_disease', plot_type='boxplot')


# ---------------------------------------------------------------------------
# R test: boxplot with zeros-heavy variable
# ---------------------------------------------------------------------------
def test_plotar_zeros_heavy_boxplot(heart_disease):
    plotar(data=heart_disease, input='num_vessels_flour',
           target='has_heart_disease', plot_type='boxplot')


# ---------------------------------------------------------------------------
# R test: filtering zeros before plotting
# ---------------------------------------------------------------------------
def test_plotar_filtered_zeros(heart_disease):
    sub = heart_disease[heart_disease['num_vessels_flour'] != 0]
    plotar(data=sub, input='num_vessels_flour',
           target='has_heart_disease', plot_type='boxplot')


# ---------------------------------------------------------------------------
# R test: ALL vars, boxplot
# ---------------------------------------------------------------------------
def test_plotar_all_vars_boxplot(heart_disease):
    num_cols = heart_disease.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'has_heart_disease']
    plotar(data=heart_disease, input=num_cols,
           target='has_heart_disease', plot_type='boxplot')


# ---------------------------------------------------------------------------
# R test: ALL vars, histdens
# ---------------------------------------------------------------------------
def test_plotar_all_vars_histdens(heart_disease):
    num_cols = heart_disease.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'has_heart_disease']
    plotar(data=heart_disease, input=num_cols,
           target='has_heart_disease', plot_type='histdens')


# ---------------------------------------------------------------------------
# R test: target as numeric
# ---------------------------------------------------------------------------
def test_plotar_numeric_target(heart_disease):
    heart_disease['has_heart_disease_num'] = (
        heart_disease['has_heart_disease'] == 'yes'
    ).astype(int)
    plotar(data=heart_disease, input='age',
           target='has_heart_disease_num', plot_type='histdens')


# ---------------------------------------------------------------------------
# Multiple inputs as list
# ---------------------------------------------------------------------------
def test_plotar_multiple_inputs(heart_disease):
    plotar(data=heart_disease, input=['age', 'max_heart_rate'],
           target='has_heart_disease', plot_type='boxplot')
