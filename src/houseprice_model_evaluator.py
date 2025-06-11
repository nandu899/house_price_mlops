from sklearn.metrics import mean_squared_error, r2_score

class HousePriceModelEvaluator:
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on test data and return MSE, R2 score, and predictions.
        """
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2, predictions
