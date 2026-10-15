"""
Fine-tune the trained model on your own webcam images (from collect_data.py),
mixed with part of the original dataset so it does not forget other letters.

Usage:
  python finetune.py --model sign_model.keras --custom custom_data --csv ..\\data\\sign_mnist_train.csv
Output:
  sign_model_custom.keras  (use this with realtime.py)
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXY"
IMG_SIZE = 28


def preprocess(roi):
    # Identical to realtime.py
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray.astype("float32") / 255.0


def load_custom(folder):
    xs, ys = [], []
    for letter in sorted(os.listdir(folder)):
        path = os.path.join(folder, letter)
        if not os.path.isdir(path) or letter.upper() not in LETTERS:
            continue
        files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png"))]
        for f in files:
            img = cv2.imread(os.path.join(path, f))
            if img is None:
                continue
            xs.append(preprocess(img))
            ys.append(LETTERS.index(letter.upper()))
        print(f"{letter}: {len(files)} images")
    x = np.array(xs, dtype="float32").reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return x, np.array(ys, dtype="int32")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sign_model.keras")
    p.add_argument("--custom", default="custom_data")
    p.add_argument("--csv", default=os.path.join("..", "data", "sign_mnist_train.csv"))
    p.add_argument("--csv_samples", type=int, default=6000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", default="sign_model_custom.keras")
    args = p.parse_args()

    xc, yc = load_custom(args.custom)
    if len(xc) == 0:
        raise SystemExit("No custom images found. Run collect_data.py first.")
    print(f"Custom images: {len(xc)}")

    df = pd.read_csv(args.csv).sample(n=args.csv_samples, random_state=0)
    yo = df["label"].values.astype("int32")
    xo = (df.drop(columns=["label"]).values.astype("float32") / 255.0).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Repeat custom data so it has a strong influence
    x = np.concatenate([xo] + [xc] * 3)
    y = np.concatenate([yo] + [yc] * 3)
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]

    model = tf.keras.models.load_model(args.model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x, y,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=64,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    model.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()