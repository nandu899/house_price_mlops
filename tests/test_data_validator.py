import pandas as pd
import pytest
from src.data_validator import DataValidator

def test_validate_passes_for_valid_data():
    df = pd.DataFrame({
        'area_sqr_ft': [1000, 1500, 2000],
        'price_lakhs': [100, 150, 200]
    })
    validator = DataValidator()
    validator.validate(df)  # Should not raise an error

def test_validate_raises_for_missing_columns():
    df = pd.DataFrame({
        'area_sqr_ft': [1000, 1500, 2000]
        # Missing 'price_lakhs'
    })
    validator = DataValidator()
    with pytest.raises(ValueError, match="Missing required columns"):
        validator.validate(df)

def test_validate_raises_for_missing_values():
    df = pd.DataFrame({
        'area_sqr_ft': [1000, None, 2000],
        'price_lakhs': [100, 150, 200]
    })
    validator = DataValidator()
    with pytest.raises(ValueError, match="Data contains missing values."):
        validator.validate(df)

def test_validate_raises_for_non_numeric_column():
    df = pd.DataFrame({
        'area_sqr_ft': [1000, 1500, 2000],
        'price_lakhs': ['one hundred', 'one fifty', 'two hundred']
    })
    validator = DataValidator()
    with pytest.raises(ValueError, match="must be numeric"):
        validator.validate(df)
