import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


def main():
    train = np.loadtxt("handout/train_triplets.txt", dtype=str, delimiter=" ")

    features_file_name = "features_vgg16.pkl"
    with open(features_file_name, 'rb') as f:
        features_dct = pkl.load(f)

    full_rows = []
    for row in train:
        full_rows.append(np.hstack([features_dct[n] for n in row]))
    train_man = np.vstack(full_rows)

    if len(train)%2 != 0:
        idx = int((len(train_man) + 1) / 2)
    else:
        idx = int(len(train_man) / 2)

    train_man_one = train_man[:idx]
    train_man_two = train_man[idx:]

    sep = int(train_man.shape[1]/3)
    t = np.copy(train_man_two[:, sep:(2*sep)])
    train_man_two[:, sep:(2*sep)] = train_man_two[:, (2*sep):]
    train_man_two[:, (2*sep):] = t

    y_labels_one = np.ones(len(train_man_one))
    y_labels_two = np.zeros(len(train_man_two))
    y_labels = np.hstack([y_labels_one, y_labels_two])
    x_train = np.vstack([train_man_one, train_man_two])

    X_train, X_test, y_train, y_test = train_test_split(x_train, y_labels, test_size=0.2, random_state=42)

    steps = [('scaling', StandardScaler()), ('reduce_dim', PCA(n_components=80)), ('clf', KNeighborsClassifier(n_neighbors=11))]
    pipeline = Pipeline(steps)

    print("Fitting the pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    test = np.loadtxt("handout/test_triplets.txt", dtype=str, delimiter=" ")

    full_rows = []
    for row in test:
        full_rows.append(np.hstack([features_dct[n] for n in row]))
    test_arr = np.vstack(full_rows)

    print("Predicting...")
    predictions_int = pipeline.predict(test_arr)

    predictions_int = predictions_int.astype(int)

    prediction_path = "predictions/prediction_pipeline_knn.txt"
    print(f"Writing predictions to {prediction_path}")
    np.savetxt(prediction_path, predictions_int, fmt="%i")


if __name__ == "__main__":
    main()