"""
Train a CNN on any image dataset organised as one folder per class, e.g.
the Kaggle "ASL Alphabet" dataset (grassknoted/asl-alphabet):

    asl_alphabet_train/
        A/  B/  C/ ... Z/  del/  nothing/  space/

Usage:
  python train_folder.py --data_dir ..\\asl_alphabet_train --limit 1000 --epochs 8

--limit  max images per class (use 500-1000 for a fast run on CPU, 0 = use all)
Outputs:
  sign_model_folder.keras and class_names.json (both needed by realtime_folder.py)
"""
import argparse
import json
import os
import random

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_root(path):
    """Descend through wrapper folders (Kaggle zips often nest the data twice)."""
    while True:
        subs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        has_images = any(f.lower().endswith(EXTS) for f in os.listdir(path))
        if len(subs) == 1 and not has_images:
            path = os.path.join(path, subs[0])
        else:
            return path


def collect(root, limit, seed=0):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    rng = random.Random(seed)
    items = []
    for i, c in enumerate(classes):
        files = [f for f in os.listdir(os.path.join(root, c)) if f.lower().endswith(EXTS)]
        rng.shuffle(files)
        if limit:
            files = files[:limit]
        items += [(os.path.join(root, c, f), i) for f in files]
        print(f"{c}: {len(files)} images")
    rng.shuffle(items)
    return items, classes


def make_ds(items, img_size, batch, training):
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(items), reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img.set_shape((img_size, img_size, 3))
        return img, label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes, img_size):
    # Inputs are raw 0-255 RGB; scaling happens inside the model.
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = models.Sequential(
        [
            layers.RandomFlip("horizontal"),  # lets left/right hands and mirrored webcams work
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.25),
        ],
        name="augmentation",
    )(x)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="sign_folder_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--out", default="sign_model_folder.keras")
    args = p.parse_args()

    root = find_root(args.data_dir)
    print(f"Using dataset folder: {root}")
    items, classes = collect(root, args.limit)
    if not items:
        raise SystemExit("No images found. Check --data_dir.")
    print(f"{len(classes)} classes, {len(items)} images total")

    n_val = int(len(items) * args.val_frac)
    train_ds = make_ds(items[n_val:], args.img_size, args.batch_size, True)
    val_ds = make_ds(items[:n_val], args.img_size, args.batch_size, False)

    with open("class_names.json", "w") as f:
        json.dump(classes, f)

    model = build_model(len(classes), args.img_size)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5),
        ],
    )
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {acc:.4f}")
    model.save(args.out)
    print(f"Saved {args.out} and class_names.json")


if __name__ == "__main__":
    main()
