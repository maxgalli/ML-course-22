import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import csv


def custom_func(x, *ws):
    w0_4 = ws[0:5]
    w5_9 = ws[5:10]
    w10_14 = ws[10:15]
    w15_19 = ws[15:20]
    w21 = ws[20]

    linear = x
    quadratic = x ** 2
    exponential = np.exp(x)
    cosine = np.cos(x)
    constant = 1

    return np.sum(
        w0_4 * linear
        + w5_9 * quadratic
        + w10_14 * exponential
        + w15_19 * cosine
        + w21 * constant,
        axis=-1,
    )


class CustomRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.ws_ = curve_fit(custom_func, X, y, p0=np.ones(21))[0]
        return self

    def predict(self, X):
        check_is_fitted(self, "ws_")
        X = check_array(X)
        return custom_func(X, *self.ws_)


def main():
    train_df = pd.read_csv("handout/train.csv")
    X = train_df.iloc[:, 2:].values
    y = train_df.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    regressor = CustomRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    error = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"RMSE: {error}")

    # dump values to csv
    with open("task1b_masgalli.csv", "w") as f:
        writer = csv.writer(f)
        for rms in regressor.ws_:
            writer.writerow([rms])


if __name__ == "__main__":
    main()
