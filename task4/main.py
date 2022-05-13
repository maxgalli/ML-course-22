import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import LassoCV

import tensorflow as tf
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            Input(shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ELumon(Model):
    def __init__(self, autoencoder):
        super(ELumon, self).__init__()
        self.encoder = autoencoder.encoder
        self.regressor = tf.keras.Sequential([
            Input(shape=self.encoder.output_shape[1:]),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
        ])
        self.top = tf.keras.Sequential([
            Input(shape=self.regressor.output_shape[1:]),
            layers.Dense(1, activation='linear', kernel_regularizer="l2")
        ])
        # Set all layers to trainable
        for layer in self.encoder.layers:
            layer.trainable = True

    def call(self, x):
        encoded = self.encoder(x)
        reg = self.regressor(encoded)
        return self.top(reg)


def main():
    pretrain_features = pd.read_csv('handout/pretrain_features.csv')
    pretrain_labels = pd.read_csv('handout/pretrain_labels.csv')
    train_features = pd.read_csv('handout/train_features.csv')
    train_labels = pd.read_csv('handout/train_labels.csv')
    test_features = pd.read_csv('handout/test_features.csv')

    # Cleaning data
    for c in pretrain_features.columns[2:]:
        if len(pretrain_features[c].unique()) < 2:
            pretrain_features = pretrain_features.drop(c, axis=1)
            train_features = train_features.drop(c, axis=1)
            test_features = test_features.drop(c, axis=1)

    # Train autoencoder
    print('Training autoencoder...')
    X_pretrain_train, X_pretrain_val, y_pretrain_train, y_pretrain_val = train_test_split(
        pretrain_features[pretrain_features.columns[2:]], pretrain_labels[["lumo_energy"]], test_size=0.001, random_state=42
    )

    autoencoder = Autoencoder(X_pretrain_train.shape[1], latent_dim=32)
    autoencoder.compile(optimizer='adam', loss=losses.MSE)

    history = autoencoder.fit(X_pretrain_train, X_pretrain_train, epochs=13, batch_size=32, validation_data=(X_pretrain_val, X_pretrain_val))

    # Train ELumon
    print('Training ELumon...')
    elumon = ELumon(autoencoder)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    elumon.compile(optimizer=opt, loss="mean_squared_error")

    X_pretrain_train, X_pretrain_val, y_pretrain_train, y_pretrain_val = train_test_split(
        pretrain_features[pretrain_features.columns[2:]], pretrain_labels[["lumo_energy"]], test_size=0.1, random_state=42
    )

    history_elumon = elumon.fit(X_pretrain_train, y_pretrain_train, epochs=20, batch_size=32, validation_data=(X_pretrain_val, y_pretrain_val))

    # Create train dataset for lasso 
    print("Training lasso...")
    X_test = test_features[test_features.columns[2:]].values
    X_train_train = train_features[train_features.columns[2:]]
    y_train_train = train_labels[["homo_lumo_gap"]]

    encoded_X_train_train = elumon.encoder.predict(X_train_train)
    elumo_X_train_train = elumon.predict(X_train_train)
    final_X_train_train = np.concatenate((encoded_X_train_train, elumo_X_train_train), axis=1)

    encoded_X_test = elumon.encoder.predict(X_test)
    elumo_X_test = elumon.predict(X_test)
    final_X_test = np.concatenate((encoded_X_test, elumo_X_test), axis=1)

    cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
    regressor = LassoCV(cv=cv)
    regressor.fit(final_X_train_train, y_train_train.values.ravel())

    y_test_pred = regressor.predict(final_X_test)
    output_df = pd.DataFrame({"Id": test_features["Id"], "y": y_test_pred})
    output_df.to_csv("submission_idi.csv", index=False)


if __name__ == "__main__":
    main()