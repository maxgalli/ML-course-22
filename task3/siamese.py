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
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet


def preprocess_image(filename, target_shape=(200, 200)):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


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


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


def main():
    # Check if running on GPU
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found')
    else:
        print('Found GPU at: {}'.format(device_name))

    # Create model
    # This is done both for train and test because I found problems in loading after saving it
    target_shape = (200, 200)

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
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )
    
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))
   
    checkpoint_dir = "checkpoints/"
    
    print("Training...")
    train = np.loadtxt("handout/train_triplets.txt", dtype=str, delimiter=" ")
    
    anchor_images = [f"handout/food/{number}.jpg" for number in list(train[:, 0])]
    positive_images = [f"handout/food/{number}.jpg" for number in list(train[:, 1])]
    negative_images = [f"handout/food/{number}.jpg" for number in list(train[:, 2])]

    image_count = len(anchor_images)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)

    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    train_dataset = train_dataset.prefetch(8)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    val_dataset = val_dataset.prefetch(8)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    history = siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[model_checkpoint_callback])
    
    print("Testing...")

    test = np.loadtxt("handout/test_triplets.txt", dtype=str, delimiter=" ")
    test_anchor_images = [f"handout/food/{number}.jpg" for number in list(test[:, 0])]
    test_positive_images = [f"handout/food/{number}.jpg" for number in list(test[:, 1])]
    test_negative_images = [f"handout/food/{number}.jpg" for number in list(test[:, 2])]
    
    test_anchor_dataset = tf.data.Dataset.from_tensor_slices(test_anchor_images)
    test_positive_dataset = tf.data.Dataset.from_tensor_slices(test_positive_images)
    test_negative_dataset = tf.data.Dataset.from_tensor_slices(test_negative_images)

    test_anchor_dataset = test_anchor_dataset.map(preprocess_image)
    test_positive_dataset = test_positive_dataset.map(preprocess_image)
    test_negative_dataset = test_negative_dataset.map(preprocess_image)

    print("Now predicting...")
    predictions = siamese_model.predict([
        np.array(list(test_anchor_dataset.as_numpy_iterator())), 
        np.array(list(test_positive_dataset.as_numpy_iterator())), 
        np.array(list(test_negative_dataset.as_numpy_iterator()))
        ])

    booleans = predictions[0] < predictions[1]
    result = booleans.astype(int)

    model_extension = datetime.today().strftime("%Y%m%d-%H%M%S")
    prediction_path = "predictions/prediction_{}.txt".format(model_extension)

    with open(prediction_path, "w") as f:
        for i in result:
            f.write(str(i) + "\n")


if __name__ == "__main__":
    main()