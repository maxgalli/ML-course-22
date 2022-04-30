import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime
import concurrent.futures
import pickle
from glob import glob

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

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


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
        model = TrainedModel(include_top=False, input_tensor=Input(shape=input_shape))
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

    anchor_train = np.vstack([features_dct[e[0]] for e in X_train])
    positive_train = np.vstack([features_dct[e[1]] for e in X_train])
    negative_train = np.vstack([features_dct[e[2]] for e in X_train])

    anchor_val = np.vstack([features_dct[e[0]] for e in X_val])
    positive_val = np.vstack([features_dct[e[1]] for e in X_val])
    negative_val = np.vstack([features_dct[e[2]] for e in X_val])

    # Model
    anchor_input = Input(shape=anchor_train.shape[1])
    positive_input = Input(shape=positive_train.shape[1])
    negative_input = Input(shape=negative_train.shape[1])

    distances = DistanceLayer()(
        anchor_input,
        positive_input,
        negative_input,
    )

    sum_ap = layers.concatenate([anchor_input, positive_input])
    sum_ap = layers.BatchNormalization()(sum_ap)
    layer_ap = layers.Dense(512, activation="relu", name="ap1")(sum_ap)
    layer_ap = layers.Dense(258, activation="relu", name="ap2")(layer_ap)
    layer_ap = layers.Dense(100, activation="relu", name="ap3")(layer_ap)

    sum_an = layers.concatenate([anchor_input, negative_input])
    sum_an = layers.BatchNormalization()(sum_an)
    layer_an = layers.Dense(512, activation="relu", name="an1")(sum_an)
    layer_an = layers.Dense(258, activation="relu", name="an2")(layer_an)
    layer_an = layers.Dense(100, activation="relu", name="an3")(layer_an)

    sum_apn = layers.concatenate([layer_ap, layer_an])
    sum_apn = layers.BatchNormalization()(sum_apn)
    layer_apn = layers.Dense(1000, activation="relu", name="apn1")(sum_apn)
    layer_apn = layers.Dense(500, activation="relu", name="apn2")(layer_apn)
    layer_apn = layers.Dense(200, activation="relu", name="apn3")(layer_apn)
    output_final = layers.Dense(1, activation="sigmoid", name="output_final")(layer_apn)

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=output_final)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit([anchor_train, positive_train, negative_train], y_train, epochs=epochs, validation_data=([anchor_val, positive_val, negative_val], y_val))

    model_extension = datetime.today().strftime("%Y%m%d-%H%M%S")
    print(f"Saving model to model/model_{model_extension}")
    model.save("models/model_{}".format(model_extension))
    with open("train_logs/history_{}.pickle".format(model_extension), 'wb') as f:
        pickle.dump(history.history, f)
