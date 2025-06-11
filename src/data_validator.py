import pandas as pd

class DataValidator:


    def __init__(self, required_columns=None):
        if required_columns is None:
            self.required_columns = ['area_sqr_ft', 'price_lakhs']
        else:
            self.required_columns = required_columns

    def validate(self, df: pd.DataFrame):
        """
        Validates the input DataFrame:
        - Checks required columns
        - Checks for missing values
        - Checks data types (numeric)
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df.isnull().values.any():
            raise ValueError("Data contains missing values.")

        for col in self.required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric.")

        print("✅ Data validation passed!")