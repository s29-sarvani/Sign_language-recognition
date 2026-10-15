"""
Real-time sign language recognition from a webcam.

Place your hand inside the green box. Keys:
  q  quit
  c  clear the typed text
  SPACE  append the current prediction to the text
  BACKSPACE  delete last character

Usage:
  python realtime.py --model sign_model.keras
"""
import argparse
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

# Index -> letter (index 9 = J and 25 = Z never occur in the dataset)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXY"
IMG_SIZE = 28


def preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = gray.astype("float32") / 255.0
    return x.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sign_model.keras")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.6, help="min confidence to show a letter")
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    history = deque(maxlen=10)  # smooth predictions over recent frames
    text = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Square region of interest on the right side of the frame
        size = int(min(h, w) * 0.55)
        x1, y1 = w - size - 30, 60
        x2, y2 = x1 + size, y1 + size
        roi = frame[y1:y2, x1:x2]

        probs = model.predict(preprocess(roi), verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        history.append(idx if conf >= args.conf else -1)

        voted, count = Counter(history).most_common(1)[0]
        current = LETTERS[voted] if voted >= 0 and count >= 6 else "-"

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
        elif key == 32 and current != "-":  # space
            text += current
        elif key in (8, 127):  # backspace
            text = text[:-1]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()