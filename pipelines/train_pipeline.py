import os
import json
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.houseprice_model_builder import HousePriceModelBuilder
from src.houseprice_model_trainer import HousePriceModelTrainer
from src.houseprice_model_evaluator import HousePriceModelEvaluator
from src.data_validator import DataValidator

def run_training_pipeline(data_path: str, model_save_path: str, evaluation_save_path: str, history_save_path: str):
    """
    Run the entire training pipeline:
    - Load data
    - Load or train model
    - Save model if newly trained
    - Evaluate model
    - Save evaluation results
    - Log performance history
    """
    # Ensure data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load dataset
    df = pd.read_csv(data_path)
    X = df[['area_sqr_ft']]
    y = df['price_lakhs']


    # Validate data
    validator = DataValidator()
    validator.validate(df)

    # Random seed for different splits (optional)
    import random
    random_seed = random.randint(0, 10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Initialize classes
    builder = HousePriceModelBuilder()
    trainer = HousePriceModelTrainer()
    evaluator = HousePriceModelEvaluator()

    # Load model if exists
    if os.path.exists(model_save_path):
        trained_model = joblib.load(model_save_path)
        print(f"🔄 Loaded existing model from {model_save_path}")
    else:
        # Build and train model
        model = builder.build_model()
        trained_model = trainer.train_model(model, X_train, y_train)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(trained_model, model_save_path)
        print(f"✅ New model trained and saved to {model_save_path}")

    # Evaluate model
    mse, r2, predictions = evaluator.evaluate_model(trained_model, X_test, y_test)

    # Save evaluation metrics
    os.makedirs(os.path.dirname(evaluation_save_path), exist_ok=True)
    evaluation_results = {
        "mean_squared_error": mse,
        "r2_score": r2
    }
    with open(evaluation_save_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"✅ Evaluation metrics saved to {evaluation_save_path}")

    # Print results
    print("\n📊 Evaluation Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # Append to performance history
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_row = {
        'timestamp': timestamp,
        'mean_squared_error': mse,
        'r2_score': r2
    }
    if os.path.exists(history_save_path):
        history_df = pd.read_csv(history_save_path)
        history_df = pd.concat([history_df, pd.DataFrame([history_row])], ignore_index=True)
    else:
        history_df = pd.DataFrame([history_row])
    history_df.to_csv(history_save_path, index=False)
    print(f"✅ Evaluation history updated at {history_save_path}")
