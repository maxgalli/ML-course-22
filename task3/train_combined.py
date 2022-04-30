import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import argparse
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import resnet


def preprocess_image(filename, target_shape=(224, 224)):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    image = tf.reshape(image, (1,) + target_shape + (3,))
    image = resnet.preprocess_input(image)
    return image


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


def main():
    # Check if running on GPU
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found')
    else:
        print('Found GPU at: {}'.format(device_name))

    # Data manipulation
    print('Loading and manipulating data...')
    train = np.loadtxt("handout/train_triplets.txt", dtype=str, delimiter=" ")
    train = train[:10000]

    anchor_images = [f"handout/food/{number}.jpg" for number in list(train[:, 0])]
    positive_images = [f"handout/food/{number}.jpg" for number in list(train[:, 1])]
    negative_images = [f"handout/food/{number}.jpg" for number in list(train[:, 2])]

    image_count = len(anchor_images)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

    anchor_dataset = anchor_dataset.map(preprocess_image)
    positive_dataset = positive_dataset.map(preprocess_image)
    negative_dataset = negative_dataset.map(preprocess_image)

    half = round(image_count / 2)
    one_anchor_dataset = anchor_dataset.take(half)
    two_anchor_dataset = anchor_dataset.skip(half)
    one_positive_dataset = positive_dataset.take(half)
    two_positive_dataset = positive_dataset.skip(half)
    one_negative_dataset = negative_dataset.take(half)
    two_negative_dataset = negative_dataset.skip(half)

    # The following has to be done beecause for some reason len(one_anchor_dataset) on GPU returns TypeError: object of type 'TakeDataset' has no len()
    if image_count % 2 == 0:
        len_one = half
        len_two = half
    else:
        len_one = half
        len_two = half - 1
    one_dataset = tf.data.Dataset.zip((one_anchor_dataset, one_positive_dataset, one_negative_dataset, tf.data.Dataset.from_tensor_slices(np.ones(len_one).reshape(-1, 1))))
    two_dataset = tf.data.Dataset.zip((two_anchor_dataset, two_negative_dataset, two_positive_dataset, tf.data.Dataset.from_tensor_slices(np.zeros(len_two).reshape(-1, 1))))
    dataset = one_dataset.concatenate(two_dataset)
    dataset = dataset.shuffle(buffer_size=image_count)

    train_dataset = dataset.take(round(.8 * image_count))
    validation_dataset = dataset.skip(round(.8 * image_count))

    X_train = train_dataset.map(lambda anchor, positive, negative, label: (anchor, positive, negative))
    X_validation = validation_dataset.map(lambda anchor, positive, negative, label: (anchor, positive, negative))
    y_train = train_dataset.map(lambda anchor, positive, negative, label: label)
    y_validation = validation_dataset.map(lambda anchor, positive, negative, label: label)

    train_dataset = tf.data.Dataset.zip((X_train, y_train))
    validation_dataset = tf.data.Dataset.zip((X_validation, y_validation))

    # Model
    print('Building model...')
    target_shape = (224, 224)

    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    distances = DistanceLayer()(
        embedding(anchor_input),
        embedding(positive_input),
        embedding(negative_input),
    )

    layer = layers.Dense(256, activation="relu")(tf.stack(distances, axis=1))
    layer = layers.Dense(1, activation="sigmoid", name="BinaryPrediction")(layer)

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=layer)

    # Train
    checkpoint_dir = "checkpoints/"

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Training...")
    history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset, callbacks=[model_checkpoint_callback]) 

    model_extension = datetime.today().strftime("%Y%m%d-%H%M%S")
    model.save("models/combined_{}".format(model_extension))


if __name__ == "__main__":
    main()