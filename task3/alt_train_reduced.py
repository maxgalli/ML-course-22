import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime
import concurrent.futures
import pickle
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16 as TrainedModel


def get_image_vector(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape(1, 224, 224, 3)
    return preprocess_input(img)


if __name__ == "__main__":
    # Check if running on GPU
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found')
    else:
        print('Found GPU at: {}'.format(device_name))

    train = np.loadtxt("handout/train_triplets.txt", dtype=str, delimiter=" ")

    # Constants
    img_rows = 224
    img_cols = 224
    input_shape = (img_rows, img_cols, 3)
    epochs = 10

    features_file_name = "features_vgg16.pkl"

    try:
        with open(features_file_name, 'rb') as f:
            features_dct = pickle.load(f)
    except FileNotFoundError:
        # Pre-trained model to extract features
        model = TrainedModel(include_top=False, input_tensor=Input(shape=input_shape), pooling='avg')
        flat1 = Flatten()(model.layers[-1].output)
        model = Model(inputs=model.inputs, outputs=flat1)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Preprocess images
        paths = glob('handout/food/*.jpg')
        paths.sort()

        preprocessed_images = {}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for path in paths:
                key = path.split('/')[-1].replace('.jpg', '')
                preprocessed_images[key] = executor.submit(get_image_vector, path)

        for key, value in preprocessed_images.items():
            preprocessed_images[key] = value.result()

        # Extract features
        features = model.predict(np.concatenate(list(preprocessed_images.values()), axis=0))
        features_dct = {k: v for k, v in zip(preprocessed_images.keys(), features)}

        with open(features_file_name, 'wb') as f:
            pickle.dump(features_dct, f)

    reduced_features_file_name = "features_vgg16_reduced.pkl"
    try:
        with open(reduced_features_file_name, 'rb') as f:
            reduced_features_dct = pickle.load(f)
    except FileNotFoundError:
    
        features_arr = np.array(list(features_dct.values()))

        steps = [('scaling', StandardScaler()), ('reduce_dim', PCA(n_components=100))]
        pipeline = Pipeline(steps)

        pipeline.fit(features_arr)

        reduced_features_arr = pipeline.transform(features_arr)

        reduced_features_dct = {}
        for i, key in enumerate(features_dct.keys()):
            reduced_features_dct[key] = reduced_features_arr[i]

        with open("reduced_features_vgg16.pkl", "wb") as f:
            pickle.dump(reduced_features_dct, f)

    if len(train)%2 != 0:
        idx = int((len(train) + 1) / 2)
    else:
        idx = int(len(train) / 2)

    train_one = train[:idx]
    train_two = train[idx:]

    sep = int(train.shape[1]/3)
    t = np.copy(train_two[:, sep:(2*sep)])
    train_two[:, sep:(2*sep)] = train_two[:, (2*sep):]
    train_two[:, (2*sep):] = t

    y_labels_one = np.ones(len(train_one))
    y_labels_two = np.zeros(len(train_two))
    y_labels = np.hstack([y_labels_one, y_labels_two])
    x_train = np.vstack([train_one, train_two])
    train = np.hstack([x_train, y_labels.reshape(-1, 1)])
    np.random.shuffle(train)

    X = train[:, :3]
    y = train[:, -1]
    y = np.array([int(e.split(".")[0]) for e in y])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    anchor_train = np.vstack([reduced_features_dct[e[0]] for e in X_train])
    positive_train = np.vstack([reduced_features_dct[e[1]] for e in X_train])
    negative_train = np.vstack([reduced_features_dct[e[2]] for e in X_train])

    anchor_val = np.vstack([reduced_features_dct[e[0]] for e in X_val])
    positive_val = np.vstack([reduced_features_dct[e[1]] for e in X_val])
    negative_val = np.vstack([reduced_features_dct[e[2]] for e in X_val])

    # Model
    anchor_input = Input(shape=anchor_train.shape[1])
    positive_input = Input(shape=positive_train.shape[1])
    negative_input = Input(shape=negative_train.shape[1])

    sum_ap = layers.concatenate([anchor_input, positive_input])
    sum_ap = layers.BatchNormalization()(sum_ap)
    layer_ap = layers.Dense(100, activation="relu", name="ap1")(sum_ap)
    layer_ap = layers.Dense(80, activation="relu", name="ap2")(layer_ap)
    layer_ap = layers.Dense(50, activation="relu", name="ap3")(layer_ap)

    sum_an = layers.concatenate([anchor_input, negative_input])
    sum_an = layers.BatchNormalization()(sum_an)
    layer_an = layers.Dense(100, activation="relu", name="an1")(sum_an)
    layer_an = layers.Dense(80, activation="relu", name="an2")(layer_an)
    layer_an = layers.Dense(50, activation="relu", name="an3")(layer_an)

    sum_apn = layers.concatenate([layer_ap, layer_an])
    sum_apn = layers.BatchNormalization()(sum_apn)
    layer_apn = layers.Dense(800, activation="relu", name="apn1")(sum_apn)
    layer_apn = layers.Dense(50, activation="relu", name="apn2")(layer_apn)
    layer_apn = layers.Dense(20, activation="relu", name="apn3")(layer_apn)
    output_final = layers.Dense(1, activation="sigmoid", name="output_final")(layer_apn)

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=output_final)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit([anchor_train, positive_train, negative_train], y_train, epochs=epochs, validation_data=([anchor_val, positive_val, negative_val], y_val))

    model_extension = datetime.today().strftime("%Y%m%d-%H%M%S")
    print(f"Saving model to model/model_{model_extension}")
    model.save("models/model_{}".format(model_extension))

    # Test
    test = np.loadtxt("handout/test_triplets.txt", dtype=str, delimiter=" ")
    anchor_test = np.vstack([reduced_features_dct[e[0]] for e in test])
    positive_test = np.vstack([reduced_features_dct[e[1]] for e in test])
    negative_test = np.vstack([reduced_features_dct[e[2]] for e in test])

    predictions = model.predict([anchor_test, positive_test, negative_test])

    prediction_path = "predictions/prediction_reduced_vgg16.txt"
    with open(prediction_path, "w") as f:
        for p in predictions:
            if p > 0.5:
                f.write("1")
            else:
                f.write("0")
            f.write("\n")
