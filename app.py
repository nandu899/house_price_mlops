from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

from pipelines.train_pipeline import run_training_pipeline

app = Flask(__name__)

MODEL_PATH = 'models/houseprice_model.pkl'

# Load the trained model if exists
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not trained yet.'}), 400

    data = request.get_json()
    area = data.get('area_sqr_ft')
    if area is None:
        return jsonify({'error': 'Missing area_sqr_ft'}), 400

    prediction = model.predict(pd.DataFrame({'area_sqr_ft': [area]}))
    return jsonify({'predicted_price_lakhs': round(prediction[0], 2)})

@app.route('/train', methods=['POST'])
def train():
    global model
    data_path = 'data/raw/home_prices.csv'  # or fetch from request if dynamic

    run_training_pipeline(
        data_path=data_path,
        model_save_path=MODEL_PATH,
        evaluation_save_path='evaluation/evaluation_metrics.json',
        history_save_path='evaluation/history.csv'
    )
    model = load_model()
    return jsonify({'message': 'Model retrained and loaded successfully.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
