from sklearn.linear_model import LinearRegression


class HousePriceModelBuilder:
    def build_model(self):
        """
        Create and return a Linear Regression model.
        """
        return LinearRegression()