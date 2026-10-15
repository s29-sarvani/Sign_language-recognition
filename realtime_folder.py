"""
Real-time recognition for a model trained with train_folder.py.

Keys:
  SPACE      add the current prediction to the text
  BACKSPACE  delete last character
  c          clear text
  q          quit

Usage:
  python realtime_folder.py --model sign_model_folder.keras --classes class_names.json
"""
import argparse
import json
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sign_model_folder.keras")
    p.add_argument("--classes", default="class_names.json")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.6)
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.classes) as f:
        classes = json.load(f)
    img_size = model.input_shape[1]

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    history = deque(maxlen=10)
    text = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        size = int(min(h, w) * 0.6)
        x1, y1 = w - size - 30, 60
        x2, y2 = x1 + size, y1 + size
        roi = frame[y1:y2, x1:x2]

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        x = rgb.astype("float32")[None, ...]  # raw 0-255; model rescales itself

        probs = model(x, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        history.append(idx if conf >= args.conf else -1)

        voted, count = Counter(history).most_common(1)[0]
        current = classes[voted] if voted >= 0 and count >= 6 else "-"
        if current == "nothing":
            current = "-"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Sign: {current}  ({conf*100:.0f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Text: {text}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            text = ""
        elif key in (8, 127):
            text = text[:-1]
        elif key == 32 and current != "-":
            if current == "space":
                text += " "
            elif current == "del":
                text = text[:-1]
            else:
                text += current

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
