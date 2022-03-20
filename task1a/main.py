import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import csv


def main():
    lambdas = [0.1, 1, 10, 100, 200]
    train_df = pd.read_csv("handout/train.csv", delimiter=",")
    X = train_df.drop(["y"], axis=1)
    y = train_df.y

    kf = KFold(n_splits=10, shuffle=True)

    rmses_averaged = {}  # dictionary with form {lambda: rmse_averaged}
    for l in lambdas:
        rmses = []
        for train_index, validation_index in kf.split(X):
            clf = Ridge(alpha=l)
            X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
            y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_validation)
            rmse = mean_squared_error(y_validation, y_pred) ** 0.5
            rmses.append(rmse)
        rmses_averaged[l] = np.mean(rmses)

    # dump values to csv
    with open("task1a_masgalli.csv", "w") as f:
        writer = csv.writer(f)
        for rms in rmses_averaged.values():
            writer.writerow([rms])


if __name__ == "__main__":
    main()
