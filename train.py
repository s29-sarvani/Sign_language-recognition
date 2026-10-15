"""
Train a CNN to recognise ASL alphabet signs (static letters).

Dataset: Sign Language MNIST (Kaggle)
  https://www.kaggle.com/datasets/datamunge/sign-language-mnist
Place these files in ./data/ :
  sign_mnist_train.csv
  sign_mnist_test.csv

Usage:
  python train.py --data_dir data --epochs 30
"""
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

NUM_CLASSES = 25  # labels run 0-24; J (9) and Z (25) need motion so are absent
IMG_SIZE = 28


def load_csv(path):
    df = pd.read_csv(path)
    y = df["label"].values.astype("int32")
    x = df.drop(columns=["label"]).values.astype("float32") / 255.0
    x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return x, y


def build_model():
    # Augmentation layers only run during training.
    augment = models.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = augment(inputs)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="sign_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out", default="sign_model.keras")
    args = p.parse_args()

    x_train, y_train = load_csv(os.path.join(args.data_dir, "sign_mnist_train.csv"))
    x_test, y_test = load_csv(os.path.join(args.data_dir, "sign_mnist_test.csv"))
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")

    model = build_model()
    model.summary()

    cbs = [
        callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
        callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor="val_accuracy"),
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()