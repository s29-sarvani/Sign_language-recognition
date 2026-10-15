"""
Collect your own training images from the webcam.

Keys:
  a-y      choose which letter you are recording (j is skipped)
  SPACE    start / stop automatic capture
  ESC      quit

Images are saved to custom_data/<LETTER>/ using the same green box as realtime.py.
Tip: record 150-300 images per letter, moving your hand slightly (closer, farther,
tilted, different positions in the box) while capturing.
"""
import argparse
import os
import time

import cv2

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXY"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="custom_data")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--every", type=int, default=3, help="save one frame every N frames")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    letter = None
    capturing = False
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Must match the ROI in realtime.py
        size = int(min(h, w) * 0.55)
        x1, y1 = w - size - 30, 60
        x2, y2 = x1 + size, y1 + size
        roi = frame[y1:y2, x1:x2]

        count = 0
        if letter:
            folder = os.path.join(args.out, letter)
            os.makedirs(folder, exist_ok=True)
            count = len(os.listdir(folder))
            if capturing and frame_no % args.every == 0:
                name = f"{int(time.time() * 1000)}.jpg"
                cv2.imwrite(os.path.join(folder, name), roi)
                count += 1
        frame_no += 1

        color = (0, 0, 255) if capturing else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        status = "RECORDING" if capturing else "paused"
        cv2.putText(frame, f"Letter: {letter or '-'}  Saved: {count}  [{status}]",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "a-y: letter | SPACE: start/stop | ESC: quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Collect data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (32,13):
            capturing = letter is not None and not capturing
        elif chr(key).upper() in LETTERS and chr(key).upper() != "J" and key != 255:
            letter = chr(key).upper()
            capturing = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()