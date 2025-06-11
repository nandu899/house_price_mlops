class HousePriceModelTrainer:
    def train_model(self, model, X_train, y_train):
        """
        Train the given model using the provided training data.
        """
        model.fit(X_train, y_train)
        return model
