import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from funpymodeling.target_analysis import var_rank_info


@pytest.fixture
def heart_disease():
    """Minimal heart_disease-like dataframe with binary numeric vars."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        'age': np.random.randint(30, 80, n),
        'gender': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'fasting_blood_sugar': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'exer_angina': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'chest_pain': np.random.choice([1, 2, 3, 4], n),
        'has_heart_disease': np.random.choice(['yes', 'no'], n),
    })
    # Make gender and exer_angina correlated with target
    mask_g1 = df['gender'] == 1
    df.loc[mask_g1, 'has_heart_disease'] = np.random.choice(
        ['yes', 'no'], mask_g1.sum(), p=[0.6, 0.4])
    df.loc[~mask_g1, 'has_heart_disease'] = np.random.choice(
        ['yes', 'no'], (~mask_g1).sum(), p=[0.25, 0.75])

    mask_ea = df['exer_angina'] == 1
    df.loc[mask_ea, 'has_heart_disease'] = np.random.choice(
        ['yes', 'no'], mask_ea.sum(), p=[0.7, 0.3])
    return df


def test_binary_numeric_vars_not_zero(heart_disease):
    """Regression: binary numeric vars must NOT have all-zero metrics."""
    result = var_rank_info(heart_disease, 'has_heart_disease')
    for var_name in ['gender', 'fasting_blood_sugar', 'exer_angina']:
        row = result[result['var'] == var_name]
        assert len(row) == 1, f"{var_name} not found in result"
        assert row['en'].values[0] > 0, f"{var_name} entropy should be > 0"
        # gender and exer_angina are correlated with target, so mi > 0
        if var_name in ('gender', 'exer_angina'):
            assert row['mi'].values[0] > 0, f"{var_name} mutual info should be > 0"
            assert row['ig'].values[0] > 0, f"{var_name} info gain should be > 0"
            assert row['gr'].values[0] > 0, f"{var_name} gain ratio should be > 0"


def test_result_columns(heart_disease):
    """Result must have the expected columns and be sorted by gr desc."""
    result = var_rank_info(heart_disease, 'has_heart_disease')
    assert list(result.columns) == ['var', 'en', 'mi', 'ig', 'gr']
    # Sorted descending by gr
    assert (result['gr'].diff().dropna() <= 0).all()


def test_target_excluded(heart_disease):
    """Target variable must not appear in the result."""
    result = var_rank_info(heart_disease, 'has_heart_disease')
    assert 'has_heart_disease' not in result['var'].values
