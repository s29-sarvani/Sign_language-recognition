"""
Web frontend for the sign language model trained with train_folder.py.

Run:
  pip install flask
  python app.py
Then open http://127.0.0.1:5000 in Chrome or Edge.

Keep index.html in the same folder as this file.
"""
import argparse
import base64
import json
import os
import threading

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_file

BASE = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(BASE, path)


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sign_model_folder.keras")
parser.add_argument("--classes", default="class_names.json")
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

print("Loading model...")
model = tf.keras.models.load_model(resolve(args.model))
with open(resolve(args.classes)) as f:
    classes = json.load(f)
img_size = model.input_shape[1]
print(f"Model ready: {len(classes)} classes, input size {img_size}x{img_size}")

lock = threading.Lock()
app = Flask(__name__)


@app.route("/")
def index():
    return send_file(os.path.join(BASE, "index.html"))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    b64 = data.get("image", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return jsonify(error="invalid image data"), 400

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="could not decode image"), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = rgb.astype("float32")[None, ...]  # raw 0-255; the model rescales itself

    with lock:
        probs = model(x, training=False).numpy()[0]

    top = np.argsort(probs)[::-1][:3]
    return jsonify(
        predictions=[{"label": classes[i], "confidence": float(probs[i])} for i in top]
    )


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{args.port} in Chrome or Edge")
    app.run(host="127.0.0.1", port=args.port, threaded=True)