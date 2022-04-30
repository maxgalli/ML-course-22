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

from train_combined import preprocess_image
from train_combined import DistanceLayer


def main():
    # Data manipulation
    print("Loading and manipulating data...")
    test = np.loadtxt("handout/test_triplets.txt", dtype=str, delimiter=" ")

    anchor_images = [f"handout/food/{number}.jpg" for number in list(test[:, 0])]
    positive_images = [f"handout/food/{number}.jpg" for number in list(test[:, 1])]
    negative_images = [f"handout/food/{number}.jpg" for number in list(test[:, 2])]

    image_count = len(anchor_images)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

    anchor_dataset = anchor_dataset.map(preprocess_image)
    positive_dataset = positive_dataset.map(preprocess_image)
    negative_dataset = negative_dataset.map(preprocess_image)

    # Note how it is done: the second part is left empty because we trained with X and y 
    test_dataset = tf.data.Dataset.zip(((anchor_dataset, positive_dataset, negative_dataset), ))

    # Load model
    clf_model_path = "models/combined_20220430-102811"
    model = tf.keras.models.load_model(clf_model_path, custom_objects={'DistanceLayer': DistanceLayer})

    # Predict
    print("Predicting...")
    predictions = model.predict(test_dataset)

    """
    print("Saving results...")
    prediction_path = "predictions/prediction_{}.txt".format(clf_model_path.split('_')[-1])
    # Write prediction to txt file as column of 0s and 1s
    with open(prediction_path, "w") as f:
        for p in predictions:
            if p > 0.5:
                f.write("1")
            else:
                f.write("0")
            f.write("\n")
    """


if __name__ == "__main__":
    main()