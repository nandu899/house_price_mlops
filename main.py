from pipelines.train_pipeline import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline(
        data_path='data/raw/home_prices.csv',
        model_save_path='models/houseprice_model.pkl',
        evaluation_save_path='evaluation/evaluation_metrics.json',
        history_save_path='evaluation/history.csv'
    )