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
    epochs = 5

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

    inp = Input(shape=(X_train.shape[1],))
    layer = Dense(512, activation=tf.nn.relu)(inp)
    layer = Dense(256, activation=tf.nn.relu)(layer)
    layer = Dense(256, activation=tf.nn.relu)(layer)
    layer = Dense(150, activation=tf.nn.relu)(layer)
    layer = Dense(80, activation=tf.nn.relu)(layer)
    layer = Dense(80, activation=tf.nn.relu)(layer)
    layer = Dense(80, activation=tf.nn.relu)(layer)
    layer = Dense(80, activation=tf.nn.relu)(layer)
    output = Dense(1, activation=tf.nn.sigmoid)(layer)

    cl_model = Model(inp, output)
    cl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = cl_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    model_extension = datetime.today().strftime("%Y%m%d-%H%M%S")
    print(f"Saving model to model/model_{model_extension}")
    cl_model.save("models/model_{}".format(model_extension))
    with open("train_logs/history_{}.pickle".format(model_extension), 'wb') as f:
        pickle.dump(history.history, f)
