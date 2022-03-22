import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error
import csv


def main():
    train_df = pd.read_csv("handout/train.csv")
    X = train_df.iloc[:, 2:].values
    y = train_df.iloc[:, 1].values

    # features
    X = np.concatenate(
        (X, X ** 2, np.exp(X), np.cos(X), np.ones((X.shape[0], 1))), axis=-1
    )

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # see https://machinelearningmastery.com/lasso-regression-with-python/

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
    model = LassoCV(alphas=np.linspace(0, 1, 10000), fit_intercept=False, max_iter=10000, cv=cv)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print("Best alpha:", model.alpha_)
    print("Coefficients:", model.coef_)

    # dump values to csv
    with open("task1b_masgalli.csv", "w") as f:
        writer = csv.writer(f)
        for rms in model.coef_:
            writer.writerow([rms])


if __name__ == "__main__":
    main()
